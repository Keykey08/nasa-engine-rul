# %% [markdown]
# # Project 3, Stage 1b: quick baselines on NASA's official test engines (FD001)
# WHY: check the problem is learnable before building complex models.
# Setup: RUL capped at 125; inputs = the 16 non-constant columns at each test engine's LAST cycle.
# Expected output (run on 9 Nov 2026): mean RMSE 43.1, linear 21.9, random forest 18.3.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW = ROOT / "data" / "raw"
cols = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv(RAW / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv(RAW / "test_FD001.txt", sep=r"\s+", header=None, names=cols)
true_rul = pd.read_csv(RAW / "RUL_FD001.txt", sep=r"\s+", header=None)[0].values

train["RUL"] = (train.groupby("unit")["cycle"].transform("max") - train["cycle"]).clip(upper=125)
features = [c for c in cols[2:] if train[c].std() > 1e-3]          # drop constant columns
last = test.groupby("unit").tail(1).sort_values("unit")            # latest reading per test engine

def nasa_score(pred, true):
    d = pred - true                                                # negative = early, positive = late
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

rows = []
for name, model in {"Predict the average": None,
                    "Linear regression": LinearRegression(),
                    "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)}.items():
    if model is None:
        pred = np.full(len(last), train["RUL"].mean())
    else:
        pred = model.fit(train[features], train["RUL"]).predict(last[features])
    rows.append({"model": name, "RMSE": np.sqrt(np.mean((pred - true_rul) ** 2)),
                 "NASA score": nasa_score(pred, true_rul), "share late": np.mean(pred > true_rul)})
res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv(ROOT / "reports" / "01b_baselines.csv", index=False)
