import os, pickle, datetime
import  math
import numpy as np
import pandas as pd
from tqdm import tqdm
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from langchain.vectorstores import Chroma


import argparse
import os
import yaml
import random

from torch.optim.lr_scheduler import ReduceLROnPlateau

def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.
 
    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_random_seed(37)


# ── New feature: WMAPE custom loss ──
class WMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_err = torch.abs(pred - target)
        # denom   = torch.clamp(target.sum(), min=self.eps)
 
        # To avoid the loss being abnormally amplified due to an extremely small denominator, the lower bound is adjusted from 1e-6 to 0.01.
        denom = torch.clamp(target.sum(), min=0.01)
        return abs_err.sum() / denom

###-----------------------
parser = argparse.ArgumentParser(description="Train multi‑modal model with Chroma news embeddings")
parser.add_argument(
    "--chroma-dir",
    default="news_chroma_db",
    help="Path to the persistent Chroma vectorstore directory (default: news_chroma_db)"
)


parser.add_argument(
    "--config-file",
    default="config.yaml",
    help="Training configuration file path"
)

args = parser.parse_args()
CHROMA_DIR = args.chroma_dir
CONFIG_PATH  = args.config_file


# —— Added: Load configuration —— 
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


#The 1e-4 here is simply a fallback value in case lr_finetune is not specified in config.yaml.
# It doesn't mean it will override the content written in the YAML file—it will only take effect if cfg.get("lr_finetune") returns None.
finetune        = cfg.get("finetune", False)
lr_finetune     = float(cfg.get("lr_finetune", 1e-4))
epochs_finetune = int(cfg.get("epochs_finetune", 2))
lr_retrain      = float(cfg.get("lr_retrain", 1e-3))
epochs_retrain  = int(cfg.get("epochs_retrain", 3))


# —— The existing static configurations, LR and EPOCHS will be dynamically overridden. —— 
# ── Config ──
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
finetune        = cfg.get('finetune', False)
lr_finetune     = float(cfg.get('lr_finetune', 1e-5))
epochs_finetune = int(cfg.get('epochs_finetune', 2))
lr_retrain      = float(cfg.get('lr_retrain', 5e-5))
# epochs_retrain  = int(cfg.get('epochs_retrain', 5))
 
epochs_retrain = int(cfg.get('epochs_retrain', 70))



STATIC_EMB_PATH = 'fund_static_embs.pkl'
TRAIN_SERIES    = 'train_series.csv'
TEST_SERIES     = 'test_series.csv'
MODEL_SAVE_PATH = "best_model_reg.pt"
HIST_WINDOW     = 7
FUTURE_WINDOW   = 7
TIME_FEAT_DIM   = 10   # 7 dow one-hot + 1 is_weekend + sin/cos month
BATCH_SIZE      = 32
 
LAG_DAYS          = [1, 7]
ROLLING_WINDOWS   = [7]
UV_ROLLING_DAYS   = [7]
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ── Load static embeddings ──

with open(STATIC_EMB_PATH, "rb") as f:
    fund_static = pickle.load(f)           # dict: fund_code → np.array(dim,)
token_dim = next(iter(fund_static.values())).shape[0]


# Calculate hist_dim = base daily 5 dimensions * 7 + time 10 + lag * len + rolling * len + uvrolling * len
hist_dim = (
    5*HIST_WINDOW             # Basic daily features
  + TIME_FEAT_DIM             # Time features
  + len(LAG_DAYS)             # Number of Lag days
  + len(ROLLING_WINDOWS)      # Rolling mean
  + len(UV_ROLLING_DAYS)      # UV rolling mean
)
 


# ── Temporal model: SequenceRegressor ──
#   LSTM is used to process daily basic features, and then static, news, and contextual features are fused together.
SEQ_INPUT_SIZE    = 5                     # The number of basic features per day: apply, redeem, uv1, uv2, uv3
SEQ_LEN           = HIST_WINDOW           # Sequence length
CONTEXT_DIM       = hist_dim - SEQ_INPUT_SIZE * SEQ_LEN
# STATIC_OUT_DIM    = 32
# NEWS_OUT_DIM      = 32
LSTM_HIDDEN_DIM   = 64


# D_MODEL=64; NHEAD=4; FF_DIM=128; NUM_LAYERS=2
# STATIC_DIM=32; NEWS_DIM=32

# ── Load time series ──

# ── Data loading ──
def load_series(path):
    df = pd.read_csv(path, parse_dates=['transaction_date'], dtype={'fund_code':str})
    df['fund_code'] = df['fund_code'].str.zfill(6)
    df['date'] = df.transaction_date.dt.date
    df.sort_values(['fund_code','date'], inplace=True)
    return df.set_index(['fund_code','date'])
train_df = load_series(TRAIN_SERIES)
val_df   = load_series(TEST_SERIES)
 


# ── Load news embeddings & dates ──

db   = Chroma(persist_directory=CHROMA_DIR, embedding_function=None)
coll = db._collection
raw  = coll.get(offset=0, limit=coll.count(), include=["embeddings","metadatas"])

news_embs_list = [np.array(e, dtype=np.float32) for e in raw["embeddings"]]
news_dates     = [datetime.date.fromisoformat(m["date"]) for m in raw["metadatas"]]


# ── Dataset ──
# ── Dataset: Completely follows the logic from s4_train_model_mm (17)_nbeat.py ──

class FundVolumeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # load news embeddings once

        db   = Chroma(persist_directory=CHROMA_DIR, embedding_function=None)
        coll = db._collection
        raw  = coll.get(offset=0, limit=coll.count(), include=['embeddings','metadatas'])

        self.news_embs  = np.stack(raw['embeddings'], axis=0).astype(np.float32)
        self.news_dates = [datetime.date.fromisoformat(m['date']) for m in raw['metadatas']]

        # build samples

        self.samples = []
        for code in df.index.get_level_values(0).unique():
            dates = df.loc[code].index
            for d in dates:
                if (d - datetime.timedelta(days=HIST_WINDOW) in dates
                and d + datetime.timedelta(days=FUTURE_WINDOW) in dates):
                    self.samples.append((code, d))  # :contentReference[oaicite:0]{index=0}
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        code, date = self.samples[idx]
        sub = self.df.xs(code, level=0)
 
        # 1) static embedding
        s_emb = torch.from_numpy(fund_static[code].astype(np.float32))
 
        # 2) news embedding: ssimilarity-weighted average
        ws   = date - datetime.timedelta(days=HIST_WINDOW)
        idxs = [i for i, d in enumerate(self.news_dates) if ws <= d <= date]
        if idxs:
            embs = self.news_embs[idxs]
            sims = embs @ fund_static[code]
            w    = np.exp(sims / math.sqrt(embs.shape[1]))
            w   /= w.sum()
            n_emb = torch.from_numpy((w[:,None] * embs).sum(0))
        else:
            n_emb = torch.zeros(token_dim, dtype=torch.float32)
 
        # 3) Historical basic features (apply, redeem, uv1, uv2, uv3)
        hist = []

        for i in range(HIST_WINDOW, 0, -1):
            r = sub.loc[date - datetime.timedelta(days=i)]

            hist += [
                r.apply_amt,
                r.redeem_amt,
                r.uv_key_page_1,
                r.uv_key_page_2,
                r.uv_key_page_3
            ]
 
        # 4) lag embedding feature
        for lag in LAG_DAYS:
            hist.append(
                sub['apply_amt']
                   .get(date - datetime.timedelta(days=lag), 0.0)
            )  # :contentReference[oaicite:1]{index=1}
 
        # 5) rolling & uv rolling

        for win in ROLLING_WINDOWS:
            idxs = pd.date_range(
                date - datetime.timedelta(days=win),
                date - datetime.timedelta(days=1)
            ).date
            vals = sub['apply_amt'].reindex(idxs, fill_value=0.0)
            hist.append(vals.mean())

        for win in UV_ROLLING_DAYS:
            idxs = pd.date_range(
                date - datetime.timedelta(days=win),
                date - datetime.timedelta(days=1)
            ).date

            vals = sub['uv_key_page_1'].reindex(idxs, fill_value=0.0)
            hist.append(vals.mean())  # :contentReference[oaicite:2]{index=2}
 
        # 6) time embedding feature
        dow    = date.weekday()
        onehot = [1.0 if dow == j else 0.0 for j in range(7)]
        is_wk  = 1.0 if dow >= 5 else 0.0
        sin_m  = math.sin(2*math.pi*date.month/12)
        cos_m  = math.cos(2*math.pi*date.month/12)
        hist += onehot + [is_wk, sin_m, cos_m]  # :contentReference[oaicite:3]{index=3}
        h_feats = torch.tensor(hist, dtype=torch.float32)
 
        # 7) label y: future apply & redeem
        app_fut = [
            sub.loc[date + datetime.timedelta(days=i)].apply_amt
            for i in range(1, FUTURE_WINDOW+1)
        ]

        red_fut = [
            sub.loc[date + datetime.timedelta(days=i)].redeem_amt
            for i in range(1, FUTURE_WINDOW+1)
        ]

        y = torch.tensor(app_fut + red_fut, dtype=torch.float32)
        return (s_emb, n_emb, h_feats), y  # :contentReference[oaicite:4]{index=4}


# ── N-BEATS Model definition (same as s4_train_model_mm (17)_nbeat.py) ── :contentReference[oaicite:6]{index=6}
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, theta_dim, hidden_dim, n_layers=4, drop =0.2):
        super().__init__()
        dims   = [input_dim] + [hidden_dim] * n_layers
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), 
                       nn.Dropout(p=drop)        # ← Add after each layer
                       ]
        self.mlp      = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_dim, input_dim)
        self.forecast = nn.Linear(hidden_dim, theta_dim)
 
    def forward(self, x):
        y = self.mlp(x)
        return self.backcast(y), self.forecast(y)
 
 
class NBeatsRegressor(nn.Module):
    ## add drop -> dropout
    def __init__(self, input_dim, theta_dim,
                 hidden_dim=256, n_blocks=3, n_layers=4, drop=0.1):
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
        for b in self.blocks:
            backcast, fc = b(resid)
            resid        = resid - backcast
            forecast    += fc
        return forecast
 
 
# ── Bi-LSTM + N-BEATS hybrid model ──
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
                 drop_rate: float = 0.05): #← 新增 drop_rate 参数
        super().__init__()
        self.seq_input_size = seq_input_size
        self.hist_window    = hist_window
        # Bi‑LSTM for history sequence
        self.bilstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  #  ← RNNs have built-in dropout, which is effective across multiple layers in the last layer; a single layer can leave 0.
        )

        # Dropout on Bi-LSTM output
        self.drop1 = nn.Dropout(p=drop_rate)

        # compute context_dim = hist_dim – seq_input_size*hist_window
        hist_dim    = (seq_input_size * hist_window
                      + TIME_FEAT_DIM
                      + len(LAG_DAYS)
                      + len(ROLLING_WINDOWS)
                      + len(UV_ROLLING_DAYS))
        context_dim = hist_dim - seq_input_size * hist_window
 
        # N-BEATS input dim includes Bi-LSTM(2*H) + context + static + news
        nbeats_input_dim = (
            2*lstm_hidden_dim +
            context_dim +
            static_dim +
            news_dim
        )
        self.nbeats = NBeatsRegressor(
            input_dim  = nbeats_input_dim,
            theta_dim  = theta_dim,
            hidden_dim = nbeats_hidden_dim,
            n_blocks   = nbeats_blocks,
            n_layers   = nbeats_layers,
            drop       = drop_rate        # ← 传入 drop_rate
        )

        # Add another Dropout before N-BEATS input.
        self.drop2 = nn.Dropout(p=drop_rate)

    def forward(self, s_emb, n_emb, h_feats):
        B = h_feats.size(0)
        # split history seq vs context
        seq_flat      = h_feats[:, : self.seq_input_size * self.hist_window]
        context_feats = h_feats[:, self.seq_input_size * self.hist_window:]
        # reshape and Bi‑LSTM encode
        seq = seq_flat.view(B, self.hist_window, self.seq_input_size)
        H, _ = self.bilstm(seq)              # (B, L, 2*lstm_hidden_dim)
        h_pool = H.mean(dim=1)               # (B, 2*lstm_hidden_dim)

        h_pool = self.drop1(h_pool)      # ← Dropout

        # concat and predict
        h_cat  = torch.cat([h_pool, context_feats], dim=1)

        h_cat  = self.drop2(h_cat)       # ← Dropout

        # 3) N-BEATS Prediction
        preds  = self.nbeats(s_emb, n_emb, h_cat)
        return preds


# ── Time series model ──
# ── Training & Validation ──

if __name__ == "__main__":
    train_ds=FundVolumeDataset(train_df)
    val_ds   = FundVolumeDataset(val_df)

    tl=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    vl=DataLoader(val_ds,batch_size=BATCH_SIZE)

    total_steps  = len(tl) * epochs_retrain
    warmup_steps = int(cfg.get('warmup_steps', total_steps // 5))

    # ---- 4.2 Training & Scheduling ----

    # Assuming train_loader is a DataLoader, it returns (s,n,h), y
    # Model & Optimizer & Scheduler
    # Dynamically obtaining dimensions
    sample_s, sample_n, sample_h = next(iter(DataLoader(train_ds,1)))[0]

    # Replace the original model definition in the training script:
    # Obtain sample dimensions

    static_dim = sample_s.size(1)
    news_dim   = sample_n.size(1)
    # no more hist_dim
    theta_dim  = FUTURE_WINDOW * 2
    
    model = BiLSTM_NBeatsHybrid(
        seq_input_size   = SEQ_INPUT_SIZE,    # e.g. 5
        hist_window      = HIST_WINDOW,       # same HIST_WINDOW you use in your Dataset
        static_dim       = static_dim,
        news_dim         = news_dim,
        theta_dim        = theta_dim,
        lstm_hidden_dim  = LSTM_HIDDEN_DIM,   # e.g. 64 or 128
        nbeats_hidden_dim= 256,
        nbeats_blocks    = 3,
        nbeats_layers    = 4
    ).to(DEVICE)

    # Based on fine-tuning, checkpoint loading and training hyperparameters are determined

    if finetune and os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH)) 
        lr,epochs=lr_finetune,epochs_finetune
        print('--- finetune -----------------')
    else:
        # Training from scratch
        lr,epochs=lr_retrain,epochs_retrain
        print('--- train from beginning -----------------')

    ##---------------------------------

    #weight_decay=1e-4   # ← Strengthening L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # If ValLoss (or ValWMAPE) does not decrease within 'patience' epochs, then set lr *= factor # to relax Patience or moderate Factor.
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer,
        mode='min', factor=0.8, patience=10, min_lr=1e-5, verbose=True
    )
    criterion = WMAPELoss()

    # -- can Switch to WMAPELoss & ReduceLROnPlateau
    # Training with MAE is more stable
    # train_loss_fn = nn.L1Loss()
    train_loss_fn = WMAPELoss(eps=0.01)

    best_val = float("inf")

    # for (s,n,h), y in tl:
    #     print("s", torch.isnan(s).any(), 
    #         "n", torch.isnan(n).any(), 
    #         "h", torch.isnan(h).any(), 
    #         "y", torch.isnan(y).any())
    #     break

    print(f'train for => {epochs} epochs')
    for epoch in range(1, epochs+1):
        # ——— Training ———
        model.train()
        train_preds, train_labels = [], []
        first = True

        for (s, n, h), y in tqdm(tl, desc=f"Train Epoch {epoch}"):
            s,n,h,y = s.to(DEVICE), n.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
            preds = model(s,n,h)
            train_preds.append(preds)
            train_labels.append(y)
            loss = train_loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
    
        train_preds  = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_wmape  = criterion(train_preds, train_labels).item()
 
        # --- Validation ---
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for (s,n,h), y in vl:
                s,n,h,y = s.to(DEVICE), n.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
                preds = model(s,n,h)
                
                val_preds.append(preds)
                val_labels.append(y)
            val_preds  = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)
            val_wmape  = criterion(val_preds, val_labels).item()
    
        # print(f"Epoch {epoch:02d}  Train WMAPE={train_wmape:.4f}  Val WMAPE={val_wmape:.4f}")
    
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}  LR={curr_lr:.2e}  Train WMAPE={train_wmape:.4f}  Val WMAPE={val_wmape:.4f}")

        # Then pass val_loss to the scheduler
        # ——— Scheduling and Saving ———
        scheduler.step(val_wmape)

        if val_wmape < best_val:
            best_val = val_wmape
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" → saved best_model_reg.pt")
 
    print("✅ Training finished.")


    # ── After training, run prediction script ──
    # (followed by s5_predict_mm.py)

 