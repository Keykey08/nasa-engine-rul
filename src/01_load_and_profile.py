# %% [markdown]
# # Project 3 (NASA / operations), Stage 1: load and profile C-MAPSS
#
# **Question:** from an engine's sensor readings so far, how many flight
# cycles does it have left before failure (Remaining Useful Life, RUL)?
# **Why business cares:** airlines and lessors schedule maintenance and
# spare-part logistics around this number; too early wastes money, too late
# grounds the aircraft.
#
# Data: NASA C-MAPSS (Saxena et al., 2008), US government public data. We start
# with subset FD001 (one operating condition, one fault mode) = the simplest.
# Module links: 4_01 linear regression, 5_01 RF/GBDT, 6_02 DNN, 2_04 transforms.

# %%
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW = ROOT / "data" / "raw"
cols = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
# WHY we name columns ourselves: the NASA files have NO header row, only
# space-separated numbers. readme.txt in the zip documents the order.
train = pd.read_csv(RAW / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv(RAW / "test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul_test = pd.read_csv(RAW / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
out = [f"Train: {train.shape[0]:,} rows, {train.unit.nunique()} engines | Test: {test.shape[0]:,} rows, {test.unit.nunique()} engines | RUL answers: {len(rul_test)}"]

# %% [markdown]
# ## 1. Build the target (RUL) for the training data
# In training, every engine runs until it fails, so at each cycle:
# RUL = (last cycle of that engine) - (current cycle).
# Common practice caps RUL at 125: early in life the engine is "healthy" and
# the sensors cannot tell 300 cycles left from 200. Capping is a modelling
# CHOICE to justify in the essay, not a fact.

# %%
train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
life = train.groupby("unit")["cycle"].max()
out.append(f"Engine lifetimes (cycles): min {life.min()}, median {int(life.median())}, max {life.max()}")

# %% [markdown]
# ## 2. Data quality: missing values and useless sensors
# WHY drop constant sensors: a sensor that never changes carries no
# information and only adds noise for the model.

# %%
out.append(f"Missing values: {int(train.isna().sum().sum())}")
std = train[[f"s{i}" for i in range(1, 22)] + ["set1", "set2", "set3"]].std()
flat = std[std < 1e-3].index.tolist()
out.append(f"Constant (useless) columns in FD001: {flat}")
useful = [c for c in std.index if c not in flat and c[0] == "s" and c[1:].isdigit()]
corr = train[useful + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
out.append("Correlation of each useful sensor with RUL:\n" + corr.round(2).to_string())

# %% [markdown]
# ## 3. Picture: how sensors drift as failure approaches

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, s in zip(axes, [corr.index[0], corr.index[1], corr.index[-1]]):
    for u in [1, 20, 40, 60, 80]:
        d = train[train.unit == u]; ax.plot(d["RUL"], d[s], lw=0.8)
    ax.invert_xaxis(); ax.set_title(f"sensor {s} vs cycles left"); ax.set_xlabel("RUL (cycles left)")
plt.tight_layout(); (ROOT / "reports").mkdir(exist_ok=True)
plt.savefig(ROOT / "reports" / "01_sensor_drift.png", dpi=150); plt.close()

(ROOT / "reports" / "01_profile.txt").write_text("\n".join(out)); print("\n".join(out))
