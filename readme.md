# VSN-LSTM Multi-Asset Return Predictor

A hybrid deep learning framework for predicting **t+1 daily log returns** of four major asset classes using **Variable Selection Networks (VSN)** and **Long Short-Term Memory (LSTM)**.

---

## 1. Project Overview

`IC_Quant_Project_v2.ipynb` implements a complete end-to-end quantitative pipeline:

- **Dynamic feature gating** via VSN-based Gated Residual Networks (GRN), which adaptively suppress uninformative features on a per-sample basis.
- **Temporal dependency modelling** through a stacked LSTM encoder.
- **Noise reduction** via a dual-stage feature selection process (Spearman clustering → XGBoost ranking).
- **Leak-free training** using Expanding Window Walk-Forward Validation with Purging and Embargo.
- **Automated hyperparameter search** using Optuna TPE sampler (30 trials).
- **Causal normalisation** using an expanding-window StandardScaler that prevents look-ahead bias.

### Target Assets

| Ticker | Asset Class | Description |
|---|---|---|
| `SPY` | Equity | S&P 500 ETF |
| `TLT.O` | Fixed Income | 20+ Year Treasury Bond ETF |
| `GLD` | Commodity | Gold ETF |
| `XLE` | Commodity | Energy Sector ETF |

**Prediction target**: $y_t = \ln(P_{t+1} / P_t)$ — next-day log return for each asset, trained independently.

---

## 2. Data Sources

### A. Market Data — Refinitiv Data API
- `SPY`, `TLT.O`, `GLD`, `XLE`, `.DXY` — daily OHLCV
- Fields: `TR.PriceOpen`, `TR.PriceHigh`, `TR.PriceLow`, `TR.PriceClose`, `TR.Volume`
- Date range: `2000-01-01` → present

### B. Macro & Alternative Indicators — FRED API

| FRED Series | Column Name | Transformation |
|---|---|---|
| `T10Y2Y` | `Term_Spread` | Level + First-difference |
| `BAA10Y` | `Credit_Spread` | Level + First-difference |
| `T10YIE` | `Breakeven_Inflation` | Level + First-difference |
| `VIXCLS` | `VIX` | Level + First-difference |
| `NASDAQXAU` | `Gold_Spot` | Log-return |
| `DCOILWTICO` | `WTI_Crude` | Log-return |

`.DXY` is additionally included as `DXY_logret` (log-return, non-target feature).

---

## 3. Feature Engineering

### Technical Indicators (Per Target Asset)

| Feature | Formula                                        | Purpose |
|---|------------------------------------------------|---|
| `ret_1d` | $\ln(P_t / P_{t-1})$                           | 1-day momentum |
| `ret_5d` | $\ln(P_t / P_{t-5})$                           | Weekly momentum |
| `ret_20d` | $\ln(P_t / P_{t-20})$                          | Monthly momentum |
| `macd` | $EMA_{12}(P) - EMA_{26}(P)$                    | Trend signal |
| `rsi_14` | Wilder RSI, $\alpha = 1/14$                    | Overbought/oversold |
| `vol_20d` | $\sigma_{20}(\text{ret_1d}) \times \sqrt{252}$ | Annualised volatility |

Each asset contributes 6 technical features, prefixed with the ticker name (e.g., `SPY_ret_1d`).

### Macro Features
- **Level + diff**: `Term_Spread`, `Credit_Spread`, `Breakeven_Inflation`, `VIX` — 8 features (4 level + 4 diff)
- **Log-return**: `Gold_Spot`, `WTI_Crude`, `DXY` — 3 features
- Total macro features: **11**

### Combined Feature Space (Before Selection)
Each target dataset contains ~17 features (6 technical + 11 macro) before the dual-stage selection.

---

## 4. Dual-Stage Feature Selection

Feature selection is performed **once** on the initial 5-year training period and the resulting `feat_cols` are applied globally across all walk-forward folds (no per-fold re-selection).

### Stage 1 — Redundancy Filter (Spearman + Ward Clustering)
1. Compute pairwise absolute Spearman correlation matrix.
2. Convert to distance matrix: $d_{ij} = 1 - |\rho_{ij}|$.
3. Apply Ward hierarchical linkage; cut at $1 - 0.90 = 0.10$.
4. Within each cluster, retain the single feature with the highest absolute Spearman IC against the target.

### Stage 2 — Relevance Filter (XGBoost Importance)
1. Fit a shallow XGBoost regressor (`max_depth=3`, `n_estimators=100`) on Stage 1 output.
2. Rank features by `feature_importances_` (information gain).
3. Retain **Top-K = 12** features.

---

## 5. Model Architecture

### Variable Selection Network (VSN)
Each of the 12 selected features is independently projected through its own **Gated Residual Network (GRN)**:

$$\text{GRN}(x) = \text{LayerNorm}\bigl(a \cdot \sigma(b) + \text{skip}(x)\bigr)$$

where $a, b = \text{split}(\text{gate}(\text{dropout}(\text{fc}_2(\text{ELU}(\text{fc}_1(x))))))$.

A separate GRN produces a **softmax attention weight** over the 12 feature embeddings, yielding a context vector that adaptively suppresses regime-irrelevant signals.

### LSTM Encoder
Processes the context sequence of length `LOOKBACK = 20` trading days and uses the **final hidden state** as the sequence representation.

### Linear Head
Maps the LSTM hidden state to a scalar $\hat{y}_t$ (predicted log return).

### Full Forward Pass
$$x \xrightarrow{\text{VSN}} \text{context}_{1:T} \xrightarrow{\text{LSTM}} h_T \xrightarrow{\text{Linear}} \hat{y}$$

---

## 6. Training Protocol

### Walk-Forward Validation (Expanding Window)

```
fold 1:  [─────── train (5yr) ───────][embargo][─ test 60d ─]
fold 2:  [──────── train (5yr+60d) ────────][embargo][─ test 60d ─]
fold 3:  [─────────── train (5yr+120d) ────────────][embargo][─ test 60d ─]
...
```

| Parameter | Value |
|---|---|
| `INITIAL_TRAIN_YEARS` | 5 |
| `TEST_SIZE` | 60 trading days (~1 quarter) |
| `EMBARGO` | 20 days (= LOOKBACK) |
| `LOOKBACK` | 20 days |

### Hyperparameter Optimisation (Optuna)
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Trials**: 30 (tuned on `fold[0]` only)
- **Objective**: MSE on fold-0 validation set
- **Budget per trial**: 15 epochs

| Hyperparameter | Search Space |
|---|---|
| `vsn_hidden_dim` | {16, 32, 64} |
| `lstm_hidden_dim` | {16, 32, 64, 128} |
| `lstm_num_layers` | {1, 2} |
| `lstm_dropout` | [0.0, 0.3] |
| `vsn_dropout` | [0.0, 0.3] |
| `lr` | [1e-4, 5e-3] log-uniform |
| `batch_size` | {32, 64, 128} |

### Training Details
- **Loss**: `HuberLoss(delta=0.01)` — robust to return outliers
- **Optimiser**: Adam
- **Gradient clipping**: global norm ≤ 1.0
- **Early stopping**: patience = 7 epochs
- **Max epochs**: 50
- **AMP**: bfloat16 on CUDA (`torch.amp.autocast`)
- **Kernel fusion**: `torch.compile()` on CUDA

### Causal Normalisation (Expanding Window StandardScaler)
Training samples are standardised using only **past observations** (no look-ahead):
- Samples 0 to `min_periods − 1`: warm-up, use stats of first 50 samples.
- Sample $i \geq 50$: use cumulative mean and std of $X_{0:i}$.

Test samples are scaled using the **full training set statistics**.

---

## 7. Persistence

All artefacts are saved to a timestamped subdirectory under `../model/YYYYMMDD_HHMMSS/`:

| File | Contents |
|---|---|
| `<ticker>_model.pt` | Model state-dict (PyTorch) |
| `<ticker>_feats.json` | Selected feature column names |
| `<ticker>_scaler.json` | Feature mean and std (JSON, no pickle) |
| `<ticker>_params.json` | Best Optuna hyperparameters |
| `<ticker>_preds.csv` | Out-of-sample predictions and ground truth |
| `<ticker>_fold_history.csv` | Per-fold MSE and hit ratio |
| `best_params.json` | Best params for all targets |
| `feat_cols.json` | Selected features for all targets |
| `summary.csv` | Overall walk-forward summary |

---

## 8. Evaluation

### Validation Metrics

| Metric | Description |
|---|---|
| MSE / MAE / RMSE | Regression error |
| IC (Spearman) | Rank correlation of predictions with realised returns |
| IC (Pearson) | Linear correlation of predictions with realised returns |
| Hit Ratio | Fraction of correct directional predictions |

### Backtesting
- **Signal**: $\text{position}_t = \text{sign}(\hat{y}_t)$ — long (+1) or short (−1)
- **Optional threshold**: zero out signals when $|\hat{y}_t| \leq \theta$
- **Equity curve**: $\exp\!\bigl(\sum \text{ret}\bigr)$

### Performance Metrics

| Metric | Description |
|---|---|
| Total Return | Cumulative return over the test period |
| CAGR | Compound Annual Growth Rate |
| Annualised Volatility | $\sigma \times \sqrt{252}$ |
| Sharpe Ratio | $\text{CAGR} / \sigma_{\text{ann}}$ (zero risk-free rate) |
| Sortino Ratio | CAGR / downside volatility |
| Max Drawdown | Peak-to-trough equity decline |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| Win Rate | Fraction of profitable days |

---

## 9. Inference

After training, `predict_next()` generates a t+1 log-return forecast for a single asset using the latest `LOOKBACK = 20` rows of raw (unscaled) data:

```python
run_dir = "../model/YYYYMMDD_HHMMSS"
loaded  = load_results(run_dir)

pred, feat_weights = predict_next(
    loaded["SPY"]["model"],
    loaded["SPY"]["feat_mean"],
    loaded["SPY"]["feat_std"],
    loaded["SPY"]["feats"],
    recent_data=datasets["SPY"].drop(columns=["target"]),
)
# pred         : float — predicted t+1 log return
# feat_weights : dict  — mean VSN attention weight per feature
```

---

## 10. File Structure

```
project/
├── src/
│   ├── IC_Quant_Project_v2.ipynb   # Main notebook (this version)
│   ├── readme.md
│   ├── refinitiv-data.config.json
│   └── secrets.json
├── data/
│   ├── merged_daily.csv            # Raw merged market + macro data
│   ├── SPY_dataset.csv             # Per-target feature datasets
│   ├── TLT_O_dataset.csv
│   ├── GLD_dataset.csv
│   └── XLE_dataset.csv
└── model/
    └── YYYYMMDD_HHMMSS/            # Timestamped run directory
        ├── SPY_model.pt
        ├── SPY_feats.json
        ├── SPY_scaler.json
        ├── SPY_params.json
        ├── SPY_preds.csv
        ├── SPY_fold_history.csv
        ├── best_params.json
        ├── feat_cols.json
        └── summary.csv
```

---

## 11. Requirements

```
numpy < 2.0 (for refinitiv API only)
pandas
torch
refinitiv-data
fredapi
optuna
xgboost
scikit-learn
scipy
matplotlib
tqdm
```
