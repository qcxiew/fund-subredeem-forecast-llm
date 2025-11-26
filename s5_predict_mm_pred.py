
import os
import pickle
import datetime
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from langchain.vectorstores import Chroma
import argparse
import os

parser = argparse.ArgumentParser(description="Predict with trained model and Chroma news embeddings")
parser.add_argument(
    "--chroma-dir",
    default="news_chroma_db",
    help="Path to the persistent Chroma vectorstore directory (default: news_chroma_db)"
)
args = parser.parse_args()
CHROMA_DIR = args.chroma_dir

# ── Configuration ──
STATIC_EMB_PATH  = "fund_static_embs.pkl"
TRAIN_SERIES     = "train_series.csv"
TEST_SERIES      = "test_series.csv"
# CHROMA_DIR       = "news_chroma_db"
MODEL_LOAD_PATH  = "best_model_reg.pt"
 
HIST_WINDOW      = 7
FUTURE_WINDOW    = 7
SEQ_INPUT_SIZE   = 5   # apply, redeem, uv1, uv2, uv3
TIME_FEAT_DIM    = 10  # 7 weekday one-hot + is_weekend + sin/cos month
LAG_DAYS         = [1, 7]
ROLLING_WINDOWS  = [7]
UV_ROLLING_DAYS  = [7]
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load time series ──
def load_series(path):
    df = pd.read_csv(path, parse_dates=["transaction_date"], dtype={"fund_code":str})
    df["fund_code"] = df["fund_code"].str.zfill(6)
    df["date"]      = df.transaction_date.dt.date
    return df.set_index(["fund_code","date"]).sort_index()
 
df_train = load_series(TRAIN_SERIES)
df_test  = load_series(TEST_SERIES)
df_all   = pd.concat([df_train, df_test])
last_known_date = max(df_all.index.get_level_values("date"))
 
# ── Load embeddings ──
with open(STATIC_EMB_PATH, "rb") as f:
    fund_static = pickle.load(f)    # dict: code → np.array(static_dim,)
token_dim = next(iter(fund_static.values())).shape[0]
 
# news embeddings & dates
db   = Chroma(persist_directory=CHROMA_DIR, embedding_function=None)
coll = db._collection
raw  = coll.get(offset=0, limit=coll.count(), include=["embeddings","metadatas"])
news_embs  = np.stack(raw["embeddings"], axis=0).astype(np.float32)
news_dates = [datetime.date.fromisoformat(m["date"]) for m in raw["metadatas"]]
 

# ── Model Definitions ──
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, theta_dim, hidden_dim, n_layers=4, drop=0.05):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * n_layers
        layers = []
        for i in range(n_layers):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(drop)
            ]
        self.mlp      = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_dim, input_dim)
        self.forecast = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x):
        y = self.mlp(x)
        return self.backcast(y), self.forecast(y)

class NBeatsRegressor(nn.Module):
    def __init__(self, input_dim, theta_dim,
                 hidden_dim=256, n_blocks=3, n_layers=4, drop=0.05):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, theta_dim, hidden_dim, n_layers, drop)
            for _ in range(n_blocks)
        ])

    def forward(self, s, n, h):
        x        = torch.cat([h, s, n], dim=1)
        resid    = x
        forecast = torch.zeros(
            x.size(0),
            self.blocks[0].forecast.out_features,
            device=x.device
        )
        for block in self.blocks:
            b, f     = block(resid)
            resid    = resid - b
            forecast = forecast + f
        return forecast

class BiLSTM_NBeatsHybrid(nn.Module):
    def __init__(self,
                 seq_input_size: int,
                 hist_window: int,
                 static_dim: int,
                 news_dim: int,
                 theta_dim: int,
                 lstm_hidden_dim: int = 64,
                 nbeats_hidden_dim: int = 256,
                 nbeats_blocks: int = 3,
                 nbeats_layers: int = 4,
                 drop_rate: float = 0.05):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.hist_window    = hist_window
 
        # Bi‑LSTM encoder
        self.bilstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop1 = nn.Dropout(drop_rate)

        # compute context dimension
        hist_dim    = (
            seq_input_size * hist_window
            + TIME_FEAT_DIM
            + len(LAG_DAYS)
            + len(ROLLING_WINDOWS)
            + len(UV_ROLLING_DAYS)
        )
        context_dim = hist_dim - seq_input_size * hist_window

        # N‑BEATS predictor
        nbeats_input_dim = (
            2 * lstm_hidden_dim +  # Bi‑LSTM output
            context_dim +
            static_dim +
            news_dim
        )
        self.drop2 = nn.Dropout(drop_rate)
        self.nbeats = NBeatsRegressor(
            input_dim   = nbeats_input_dim,
            theta_dim   = theta_dim,
            hidden_dim  = nbeats_hidden_dim,
            n_blocks    = nbeats_blocks,
            n_layers    = nbeats_layers,
            drop        = drop_rate
        )
 
    def forward(self, s_emb, n_emb, h_feats):
        B = h_feats.size(0)
        # split into sequence vs context
        seq_flat      = h_feats[:, : self.seq_input_size * self.hist_window]
        context_feats = h_feats[:, self.seq_input_size * self.hist_window:]
        seq = seq_flat.view(B, self.hist_window, self.seq_input_size)
 
        # Bi‑LSTM + dropout
        H, _    = self.bilstm(seq)
        h_pool  = H.mean(dim=1)
        h_pool  = self.drop1(h_pool)
 
        # concat + dropout
        h_cat   = torch.cat([h_pool, context_feats], dim=1)
        h_cat   = self.drop2(h_cat)
 
        return self.nbeats(s_emb, n_emb, h_cat)
    
# ── Instantiate & load model ──
static_dim = token_dim
news_dim   = token_dim
theta_dim  = FUTURE_WINDOW * 2
 
model = BiLSTM_NBeatsHybrid(
    seq_input_size    = SEQ_INPUT_SIZE,
    hist_window       = HIST_WINDOW,
    static_dim        = static_dim,
    news_dim          = news_dim,
    theta_dim         = theta_dim,
    lstm_hidden_dim   = 64,
    nbeats_hidden_dim = 256,
    nbeats_blocks     = 3,
    nbeats_layers     = 4,
    drop_rate         = 0.05
).to(DEVICE)
 
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
model.eval()

# ── Feature builder (unchanged) ──

def build_features(code: str, pivot_date: datetime.date):
    # 1) static embedding
    emb = fund_static[code].astype(np.float32)
    s = torch.from_numpy(emb).unsqueeze(0).to(DEVICE)
    # 2) news embedding
    ws = pivot_date - datetime.timedelta(days=HIST_WINDOW)
    idxs = [i for i, d in enumerate(news_dates) if ws <= d <= pivot_date]

    if idxs:
        slice_embs = news_embs[idxs]
        sims = slice_embs @ emb
        w = np.exp(sims / math.sqrt(token_dim))
        w /= w.sum()
        n_emb = torch.from_numpy((w[:, None] * slice_embs).sum(0))
    else:
        n_emb = torch.zeros(token_dim, dtype=np.float32)
    n = n_emb.unsqueeze(0).to(DEVICE)
 
    # 3) base history features
    hist = []
    for i in range(HIST_WINDOW, 0, -1):
        d = pivot_date - datetime.timedelta(days=i)
        if (code, d) in df_all.index:
            row = df_all.loc[(code, d)]
            hist += [
                row.apply_amt, row.redeem_amt,
                row.uv_key_page_1, row.uv_key_page_2, row.uv_key_page_3
            ]
        else:
            hist += [0.0] * 5
    code_df = df_all.xs(code, level=0)
 
    # 4) lag features
    for lag in LAG_DAYS:
        ld = pivot_date - datetime.timedelta(days=lag)
        hist.append(code_df['apply_amt'].get(ld, 0.0))
 
    # 5) rolling windows
    for win in ROLLING_WINDOWS:
        idxs = pd.date_range(
            pivot_date - datetime.timedelta(days=win),
            pivot_date - datetime.timedelta(days=1)
        ).date

        vals = code_df['apply_amt'].reindex(idxs, fill_value=0.0)
        hist.append(vals.mean())
 
    # 6) uv rolling
    for win in UV_ROLLING_DAYS:
        idxs = pd.date_range(
            pivot_date - datetime.timedelta(days=win),
            pivot_date - datetime.timedelta(days=1)
        ).date

        vals = code_df['uv_key_page_1'].reindex(idxs, fill_value=0.0)
        hist.append(vals.mean())
 
    # 7) time features
    dow = pivot_date.weekday()
    onehot = [1.0 if dow == j else 0.0 for j in range(7)]
    is_wk = 1.0 if dow >= 5 else 0.0
    sin_m = math.sin(2 * math.pi * pivot_date.month / 12)
    cos_m = math.cos(2 * math.pi * pivot_date.month / 12)
    hist += onehot + [is_wk, sin_m, cos_m]
    h_feats = torch.tensor(hist, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return s, n, h_feats


# ── Prediction loops (unchanged) ──
hist_preds, fut_preds = [], []
 
# historical backtest
for code in sorted(df_all.index.get_level_values("fund_code").unique()):
    for d in range(HIST_WINDOW,0,-1):
        target_date = last_known_date - datetime.timedelta(days=d)
        pivot_date  = target_date - datetime.timedelta(days=1)
        try:
            s,n,h = build_features(code, pivot_date)
            out   = model(s,n,h).cpu().detach().numpy().reshape(-1)
            a_pred = float(out[0])
            r_pred = float(out[FUTURE_WINDOW])
        except KeyError:
            a_pred = r_pred = np.nan

        if (code,target_date) in df_all.index:
            row = df_all.loc[(code,target_date)]
            a_act, r_act = float(row.apply_amt), float(row.redeem_amt)
        else:
            a_act = r_act = np.nan

        hist_preds.append({
            "fund_code": code,
            "transaction_date": target_date.strftime("%Y%m%d"),
            "apply_amt_pred": a_pred,
            "redeem_amt_pred": r_pred,
            "apply_amt_actual": a_act,
            "redeem_amt_actual": r_act
        })
 
# future forecast
for code in sorted(df_all.index.get_level_values("fund_code").unique()):
    pivot_date = last_known_date
    s,n,h = build_features(code, pivot_date)
    out   = model(s,n,h).cpu().detach().numpy().reshape(-1)
    for i in range(1, FUTURE_WINDOW+1):
        pred_date = last_known_date + datetime.timedelta(days=i)
        a_pred = float(out[i-1])
        r_pred = float(out[FUTURE_WINDOW + i-1])
        fut_preds.append({
            "fund_code": code,
            "transaction_date": pred_date.strftime("%Y%m%d"),
            "apply_amt_pred": a_pred,
            "redeem_amt_pred": r_pred
        })
 
# save
ts = datetime.datetime.now().strftime("%y%m%d_%H%M")
out_dir = "result_pred"
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(fut_preds).to_csv(
    os.path.join(out_dir, f"predict_result_{ts}_biLstmNbeat_dro_seed.csv"), index=False)

pd.DataFrame(hist_preds+fut_preds).to_csv(
    os.path.join(out_dir, f"predict_hisfut_{ts}_biLstmNbeat_dro_seed.csv"), index=False)

future_fname =os.path.join(out_dir, f"predict_result_{ts}.csv")
hisfut_fname = os.path.join(out_dir, f"predict_hisfut_{ts}.csv")

print(f"Saved future predictions to {future_fname}" )
print(f"Saved history+future results to {hisfut_fname}")

print("Saved predictions.")


