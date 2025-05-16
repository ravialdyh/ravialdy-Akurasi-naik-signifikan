# ravialdy-Akurasi-naik-signifikan

# ────────────────────────────────────────────────────────────────────────────────
# Section 7 – Visual deep-dive on outliers & regime stability
# ────────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Helper: ensure tight, readable titles
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11})

# 7.1  Overlaid histogram: user-fixed vs model-driven (linear & log-x)
# -------------------------------------------------------------------
fig, ax = plt.subplots()
bins = 50
ax.hist(df.loc[df.aitempchanged == 1, "label_interval"], bins=bins,
        alpha=0.5, density=True, label="user_fixed")
ax.hist(df.loc[df.aitempchanged == 0, "label_interval"], bins=bins,
        alpha=0.5, density=True, label="model_driven")
ax.set_xlabel("call interval (min)")
ax.set_ylabel("density")
ax.set_title("Distribution of call intervals – linear scale")
ax.legend()
plt.tight_layout()

fig, ax = plt.subplots()
max_val = df["label_interval"].max() + 1
log_bins = np.logspace(0, np.log10(max_val), 60)
ax.hist(df.loc[df.aitempchanged == 1, "label_interval"], bins=log_bins,
        alpha=0.5, density=True, label="user_fixed")
ax.hist(df.loc[df.aitempchanged == 0, "label_interval"], bins=log_bins,
        alpha=0.5, density=True, label="model_driven")
ax.set_xscale("log")
ax.set_xlabel("call interval (min, log scale)")
ax.set_ylabel("density")
ax.set_title("Distribution of call intervals – log-x tail view")
ax.legend()
plt.tight_layout()

# 7.2  Empirical CDFs – raw vs after cleaning
# -------------------------------------------
def plot_ecdf(ax, series, label):
    vals = np.sort(series)
    y = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, y, marker="", linestyle="-", label=label)

fig, ax = plt.subplots()
plot_ecdf(ax, df["label_interval"], "raw")
plot_ecdf(ax, df.loc[~df.const_run_flag & df.label_interval_winz.notna(),
                     "label_interval_winz"],
          "rule+winz")
ax.set_xlabel("call interval (min)")
ax.set_ylabel("empirical CDF")
ax.set_title("ECDF – effect of constant-run filter + winsorisation")
ax.legend()
plt.tight_layout()

# 7.3  Box- & violin-plots: regime-specific before/after
# ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

ax1.boxplot(
    [df.loc[df.aitempchanged == 1, "label_interval"],
     df.loc[df.aitempchanged == 0, "label_interval"]],
    labels=["user_fixed", "model_driven"])
ax1.set_title("Raw")

ax2.boxplot(
    [df.loc[(df.aitempchanged == 1) & ~df.const_run_flag &
            df.label_interval_winz.notna(), "label_interval_winz"],
     df.loc[(df.aitempchanged == 0) & ~df.const_run_flag &
            df.label_interval_winz.notna(), "label_interval_winz"]],
    labels=["user_fixed", "model_driven"])
ax2.set_title("After rule+winz")
fig.suptitle("Call-interval distribution by regime")
plt.tight_layout()

# 7.4  Proportion of rows removed by each rule (bar chart)
# --------------------------------------------------------
rules = {
    "constant_run": df.const_run_flag,
    "winsorised":   df.label_interval_winz.isna() & ~df.const_run_flag,
    "iso_outlier":  df.iso_outlier & ~df.const_run_flag & df.label_interval_winz.notna(),
}
fig, ax = plt.subplots()
heights = [mask.mean()*100 for mask in rules.values()]
ax.bar(rules.keys(), heights)
ax.set_ylabel("% of rows flagged")
ax.set_title("Row-removal impact by rule (overall)")
plt.tight_layout()

# 7.5  Tail ratio per regime (p99 / p95) – raw vs cleaned
# -------------------------------------------------------
def tail_ratio(series):
    return series.quantile(0.99) / series.quantile(0.95)

metrics = []
for name, subset in [("user_raw",  df[(df.aitempchanged == 1)]),
                     ("model_raw", df[(df.aitempchanged == 0)]),
                     ("user_clean",  df[(df.aitempchanged == 1) & ~df.const_run_flag &
                                        df.label_interval_winz.notna()]),
                     ("model_clean", df[(df.aitempchanged == 0) & ~df.const_run_flag &
                                        df.label_interval_winz.notna()])]:
    metrics.append((name, tail_ratio(subset["label_interval" if "raw" in name else
                                          "label_interval_winz"])))

fig, ax = plt.subplots()
ax.bar([m[0] for m in metrics], [m[1] for m in metrics])
ax.set_ylabel("p99 / p95 ratio")
ax.set_title("Tail heaviness – before vs after cleaning")
plt.xticks(rotation=15)
plt.tight_layout()

# 7.6  Time-series drift of high quantiles (optional if eventtime exists)
# ----------------------------------------------------------------------
if "eventtime" in df.columns:
    df_ts = df.copy()
    df_ts["date"] = pd.to_datetime(df_ts["eventtime"]).dt.date
    q_daily = (df_ts.groupby(["date", "aitempchanged"])
                     ["label_interval"]
                     .quantile([0.5, 0.9, 0.99])
                     .unstack(level=-1)
                     .unstack(level=-1)
                     .reset_index())
    # q_daily now has columns like ('label_interval', 0, 0.99) – flatten for clarity
    q_daily.columns = ["date"] + [f"chg{flag}_q{int(q*100)}"
                                  for flag, q in itertools.product([0, 1], [50, 90, 99])]

    fig, ax = plt.subplots(figsize=(9, 4))
    for col in [c for c in q_daily.columns if c != "date"]:
        ax.plot(q_daily["date"], q_daily[col], label=col)
    ax.set_ylabel("minutes")
    ax.set_title("Daily median / p90 / p99 – regime split")
    ax.legend(ncol=3, fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pd.to_datetime(x).strftime("%Y-%m-%d")))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

# 7.7  Scatter: interval vs. current temperature (colour = regime)
# ----------------------------------------------------------------
if "currentt" in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df.loc[df.aitempchanged == 1, "currentt"],
               df.loc[df.aitempchanged == 1, "label_interval"],
               alpha=0.3, label="user_fixed", s=6)
    ax.scatter(df.loc[df.aitempchanged == 0, "currentt"],
               df.loc[df.aitempchanged == 0, "label_interval"],
               alpha=0.3, label="model_driven", s=6)
    ax.set_xlabel("current indoor temp (°C)")
    ax.set_ylabel("call interval (min)")
    ax.set_title("Interval vs. current temperature")
    ax.legend()
    plt.tight_layout()

print("✅  Visualization section completed — run cells to render figures.")
