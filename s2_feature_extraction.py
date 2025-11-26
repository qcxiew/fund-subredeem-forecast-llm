# feature_extraction.py
# turn the fund basic profile info into embedding. read data from fund_details.jsonl.
# feature_extraction.py can be run only 1 time, toto get the basic profile embedding.
 
import json
import pickle
import hashlib
import datetime
 
import torch
import torch.nn.functional as F
import numpy as np
 
from transformers import AutoTokenizer, AutoModel
 
# —— Param configuration ——  

MODEL_NAME   = "Qwen/Qwen3-Embedding-4B"
# MODEL_NAME  = "/home/xxx/modelscope/Qwen3/Qwen3_06B_embed/Qwen3-Embedding-0.6B"

PERSIST_DIR   = "."           # For persistence
BATCH_SIZE    = 8             # adjustable base on GPU memory
CHUNK_SIZE    = 512           # Consistent with the news task, s3_vectorstore_build.py

# (Static descriptions typically have <512 tokens, no actual block division required)
# — Initialize the tokenizer and model —

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_emb = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    trust_remote_code=True
)

model_emb = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(device).eval()
 
def last_token_pooling(hidden_states, attention_mask):
    """Take the representation vector of the last non-PAD token in each sequence and normalize it."""
    idxs = attention_mask.sum(dim=1) - 1  # (B,)
    batch_embs = []

    for i, idx in enumerate(idxs):
        batch_embs.append(hidden_states[i, idx, :])

    emb = torch.stack(batch_embs, dim=0)   # (B, D)
    return F.normalize(emb, p=2, dim=1)    # L2 normalization

def get_embs(texts: list[str],
             batch_size: int = BATCH_SIZE,
             max_length: int = CHUNK_SIZE) -> np.ndarray:
    """
        Embedding a set of short texts (<= CHUNK_SIZE tokens):
        - Set truncation/max_length to CHUNK_SIZE
        - Last-token pooling + L2 normalization
    """
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inp = tokenizer_emb(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
 
        with torch.no_grad():
            hs = model_emb(**inp).last_hidden_state  # (B, L, D)
 
        pooled = last_token_pooling(hs, inp.attention_mask)  # (B, D)
        all_embs.append(pooled.cpu().numpy())
 
    return np.concatenate(all_embs, axis=0)  # (N, D)
 
def build_fund_static(
    input_jsonl: str = "fund_details.jsonl",
    output_pkl:  str = "fund_static_embs.pkl",
):
    """
    Read fund_details.json → Construct the description → Extract the embedding → Save.
    """

    fund_embs, codes, texts = {}, [], []
    # 1) Load and concatenate text

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            code = rec.get("fund_code") or rec.get("code")
            codes.append(code)
            parts = []

            for key in [
                "full_name", "short_name", "fund_type", "issue_date",
                "inception_date", "asset_scale", "share_scale",
                "manager", "trustee", "dividend_policy",
                "management_fee", "custody_fee", "service_fee",
                "max_subscription_fee", "max_redemption_fee",
                "benchmark", "tracking_target",
                "investment_objective", "investment_scope",
                "investment_strategy", "risk_return_profile"
            ]:

                val = rec.get(key)
                if val:
                    parts.append(f"{key.replace('_',' ')}:{val}")

            # Description of merging
            desc = ";".join(parts)
            texts.append(desc)

    # 2) generating embedding
    print(f"Start generating {len(texts)} funds' static embedding (max_length={CHUNK_SIZE}）…")
    embs = get_embs(texts)  # 输出 shape = (len(texts), 1024)
 
    # 3) Save to py dict
    for code, emb in zip(codes, embs):
        fund_embs[code] = emb
 
    # 4) Serialization save/dump to file
    with open(output_pkl, "wb") as fout:
        pickle.dump(fund_embs, fout)
 
    print(f"saved Static embedding to {output_pkl}，has {len(fund_embs)} record items。")
 
if __name__ == "__main__":
    build_fund_static()

 