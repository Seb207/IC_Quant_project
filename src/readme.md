# VSN-LSTM Multi-Asset Return Predictor

A hybrid deep learning framework for predicting $t+1$ daily log returns of multi-asset classes (Equity, Bonds, Commodities) using **Variable Selection Networks (VSN)** and **Long Short-Term Memory (LSTM)**.

## 1. Project Overview
This project is currently being developed in `src/IC_Quant_Project.ipynb` as a notebook-first pipeline.

The target design is an end-to-end quantitative workflow that:
- Captures dynamic market regimes using VSN-based feature gating.
- Learns long-term temporal dependencies through LSTM.
- Addresses high-dimensional noise and multi-collinearity via a dual-stage feature selection process.
- Strictly follows professional backtesting protocols (Expanding Window, Purging/Embargo).

### Current Status Snapshot
- **Implemented in notebook**: Data acquisition, preprocessing/transformation, technical feature engineering, dual-stage feature selection.
- **In progress / pending**: VSN embedding, LSTM training, walk-forward validation, and backtesting.

---

## 2. Data Sources & Factor Universe

### A. Target Assets (Refinitiv Data API)
- **Equity**: `SPY` (S&P 500 ETF)
- **Fixed Income**: `TLT.O` (20+ Yr Treasury Bond ETF)
- **Commodity**: `GLD` (Gold ETF), `XLE` (Energy Sector ETF)

### B. Auxiliary Market Input (Refinitiv)
- **USD Regime Proxy (non-target feature)**: `.DXY` (used as explanatory input only, not a prediction target)

### C. Macro & Alternative Indicators (FRED API)
- **Macro Factors**:
    - Term Spread: `T10Y2Y` (10Y-2Y Treasury)
    - Credit Spread: `BAA10Y` (Moody's BAA - 10Y Treasury)
    - Breakeven Inflation: `T10YIE` (10Y Breakeven)
- **Alternative & Tail-Risk Indicators**:
    - Volatility (Magnitude): `VIXCLS` (CBOE VIX)
    - Gold Spot: `NASDAQXAU` (Gold Price)
    - WTI Crude: `DCOILWTICO` (WTI Oil)

### D. Acquisition Progress (Implemented)
- Refinitiv and FRED fetch functions are operational in `src/IC_Quant_Project.ipynb`.
- Per-ticker status logging is enabled (`Fetching`, `OK`, `EMPTY`, `FAIL`) with row count and date range.
- Data alignment uses Refinitiv trading dates as the base index, with macro series merged and forward-filled.

### E. Technical Indicators (Derived from Refinitiv OHLCV)
- **Trend/Momentum**: 
    - **MACD**: $EMA_{12}(P) - EMA_{26}(P)$
    - **Returns**: 1D, 5D, and 20D Log Returns.
- **Oscillator**: 
    - **RSI (14-Day)**: Relative Strength Index.
- **Risk**: 
    - **Rolling Volatility**: 20-Day annualized standard deviation of log returns.

---

## 3. Preprocessing & Feature Engineering

### Progress Status
- **Implemented in notebook**: Macro transformation (`level` + `diff` or log return), technical indicators, and per-target dataset construction.
- **Implemented in notebook**: Data export for each target dataset into `data/`.
- **Pending hardening**: Leakage-safe scaling inside walk-forward folds.
- `DXY_logret` is currently treated as an auxiliary input feature and is not included in target prediction labels.

### A. Transformation Logic
Features are transformed based on their nature to preserve both momentum and regime information:

| Feature Type | Transformation (for Correlation) | Transformation (for Model Input) |
| :--- | :--- | :--- |
| **Asset Prices** | Log Return ($\ln(P_t/P_{t-1})$) | Log Return |
| **Spreads/VIX/Rates** | 1st-order Difference ($\Delta X_t$) | **Both `Level` AND `diff()`** |
| **Commodity Spot Series** | Log Return | Log Return |

*Note: Current notebook version tracks VIX levels/differences and commodity spot log returns; additional stress indicators can be reintroduced later if needed.*

### B. Technical Engineering Details
- **Alignment**: Align all FRED data to Refinitiv trading days using **Forward Fill (`ffill`)**.
- **Scaling**: Apply `RobustScaler` (Median/IQR) to handle fat-tails and outliers in financial time series.

---

## 4. Dual-Stage Feature Selection (Per Target)

### Progress Status
- **Implemented in notebook**: Stage 1 (Spearman + hierarchical clustering) and Stage 2 (XGBoost importance ranking).
- **Current output**: Target-wise selected feature lists.
- **Pending**: Moving selection into walk-forward folds to eliminate look-ahead bias.

### Stage 1: Redundancy Filter (Spearman + Clustering)
- **Goal**: Prevent weight dilution in VSN by removing overlapping signals (e.g., VIX vs SKEW vs Credit Spread).
- **Process**: 
    1. Calculate Spearman Rank Correlation on stationary data.
    2. Perform Hierarchical Clustering (Ward's Linkage).
    3. Retain the feature with the highest absolute IC relative to the target if Correlation $> 0.90$.

### Stage 2: Relevance Filter (XGBoost Screening)
- **Process**: 
    1. Train a shallow XGBoost Regressor on the training set.
    2. Select Top $K$ features based on **Information Gain**.

---

## 5. Model Architecture: VSN + LSTM

### Status: Planned (Not Implemented Yet)
- VSN and LSTM sections are scaffolded in the notebook, but model code is not finalized.
- Next milestone is integrating selected features into a leak-free train/validation pipeline.

### Variable Selection Network (VSN)
- Employs **Gated Residual Networks (GRN)** for each feature.
- Dynamically calculates Softmax-weighted feature importance at each timestamp $t$.

### LSTM
- Processes the gated sequence to learn non-linear temporal patterns.
- Hidden Layers and Dropout rates are optimized via the Validation set.

---

## 6. Validation & Backtesting
### Status: Planned (Not Implemented Yet)
- **Data Range**: January 2000 – Present.
- **Protocol**: **Expanding Window Walk-Forward Validation**.
- **Leakage Control**: 
    - **Purging**: Remove training samples overlapping with the test period.
    - **Embargo**: Skip samples immediately following the test period to account for autocorrelation.

Current notebook execution has not yet implemented full walk-forward training/evaluation outputs.

---

## 7. Development Roadmap
- [x] **Task 1 (Notebook)**: Data acquisition from Refinitiv + FRED with ticker-level logging.
- [x] **Task 2 (Notebook)**: Preprocessing and feature engineering (macro transforms + technical indicators).
- [x] **Task 3 (Notebook)**: Dual-stage feature selection (Spearman clustering + XGBoost ranking).
- [ ] **Task 4 (Next)**: Walk-forward, leakage-safe feature selection and scaling.
- [ ] **Task 5 (Next)**: VSN + LSTM implementation and model training.
- [ ] **Task 6 (Next)**: Validation, backtesting, and performance reporting.
