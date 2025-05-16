# ravialdy-Akurasi-naik-signifikan

```python
# ────────────────────────────────────────────────────────────────────────────────
# Visual diagnostics for “constant-run” impact in your data
# Paste this into your notebook _after_ you’ve computed `const_run_flag`
# ────────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ────────────────────────────────────────────────────────────────────────────────
# 1. Re-compute run segments (if not already in your df)
# ────────────────────────────────────────────────────────────────────────────────

# Parameters – adjust to your pipeline settings
SEQ_COLS      = ["uuid", "session_id"]   # grouping keys
ORDER_COL     = "eventtime"              # monotonic per session
TEMP_COL      = "currentt"               # optional guard
TEMP_EPS      = 0.2                      # °C tolerance
CONST_RUN_MIN = 6                        # 6+ identical logs = “artefact”

# Break condition: interval changes OR temperature changes beyond eps
def _break_condition(sub: pd.DataFrame) -> pd.Series:
    iv_change = sub["label_interval"].diff().abs() > 1e-9
    if TEMP_COL in sub:
        tmp_change = sub[TEMP_COL].diff().abs() > TEMP_EPS
        return iv_change | tmp_change
    return iv_change

# Sort and assign run IDs
df_sorted = df.sort_values(SEQ_COLS + [ORDER_COL]).copy()
mask_break = (
    df_sorted
      .groupby(SEQ_COLS, sort=False)
      .apply(_break_condition)
      .reset_index(level=0, drop=True)
)
run_id   = mask_break.cumsum()
df_sorted["run_id"] = run_id
df_sorted["run_len"] = (
    df_sorted
      .groupby("run_id")["label_interval"]
      .transform("count")
)

# Merge back to main df
df = df.merge(
    df_sorted[["run_id", "run_len"]],
    left_index=True, right_index=True, how="left"
)

# Label each segment as constant-run or not
df["const_run_flag"] = df["run_len"] >= CONST_RUN_MIN
df["regime"] = np.where(df["aitempchanged"]==1, "user_fixed", "model_driven")

# ────────────────────────────────────────────────────────────────────────────────
# 2. Compute summary metrics
# ────────────────────────────────────────────────────────────────────────────────

# 2.1 Proportion of rows flagged overall & by regime
row_props = (
    df.groupby("regime")["const_run_flag"]
      .mean()
      .rename("pct_rows_flagged")
      .mul(100)
)
total_row_prop = df["const_run_flag"].mean() * 100

# 2.2 Proportion of run-segments flagged overall & by regime
seg_df = (
    df[["run_id", "run_len", "regime", "const_run_flag"]]
      .drop_duplicates(subset="run_id")
)
run_props = (
    seg_df.groupby("regime")["const_run_flag"]
          .mean()
          .rename("pct_runs_flagged")
          .mul(100)
)
total_run_prop = seg_df["const_run_flag"].mean() * 100

# ────────────────────────────────────────────────────────────────────────────────
# 3. Visualization code (no seaborn / no custom colors)
# ────────────────────────────────────────────────────────────────────────────────

# 3.1 Bar: % rows flagged by regime
fig, ax = plt.subplots()
row_props.plot.bar(ax=ax)
ax.set_ylabel("% of rows flagged")
ax.set_title("Constant-run rows flagged, by regime")
plt.tight_layout()

# 3.2 Bar: % run-segments flagged by regime
fig, ax = plt.subplots()
run_props.plot.bar(ax=ax)
ax.set_ylabel("% of run segments flagged")
ax.set_title("Constant-run segments flagged, by regime")
plt.tight_layout()

# 3.3 Histogram: distribution of run lengths (all vs flagged)
fig, ax = plt.subplots()
bins = range(1, seg_df["run_len"].max()+2)
ax.hist(seg_df["run_len"], bins=bins, alpha=0.5, density=False, label="all runs")
ax.hist(seg_df.loc[seg_df.const_run_flag, "run_len"], bins=bins, alpha=0.5,
        density=False, label="flagged runs")
ax.set_xlabel("run length (number of logs)")
ax.set_ylabel("count of segments")
ax.set_title("Run-length distribution – all vs flagged")
ax.legend()
plt.tight_layout()

# 3.4 Boxplot: run lengths by regime, flagged vs non-flagged
fig, axes = plt.subplots(1, 2, figsize=(8,4), sharey=True)
for i, regime in enumerate(["user_fixed","model_driven"]):
    sub = seg_df[seg_df["regime"] == regime]
    axes[i].boxplot(
        [ sub.loc[~sub.const_run_flag, "run_len"],
          sub.loc[sub.const_run_flag,  "run_len"] ],
        labels=["ok","flagged"]
    )
    axes[i].set_title(f"{regime} run lengths")
axes[0].set_ylabel("run length")
plt.suptitle("Run lengths – flagged vs non-flagged by regime")
plt.tight_layout(rect=[0, 0, 1, 0.93])

# 3.5 Proportion flagged over time (daily) if eventtime exists
if ORDER_COL in df:
    df["date"] = pd.to_datetime(df[ORDER_COL]).dt.date
    daily = (
        df.groupby(["date","regime"])["const_run_flag"]
          .mean()
          .mul(100)
          .unstack()
    )
    fig, ax = plt.subplots(figsize=(9,3))
    daily.plot(ax=ax)
    ax.set_ylabel("% rows flagged")
    ax.set_title("Daily flagged-row rate per regime")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x,y: pd.to_datetime(x).strftime("%Y-%m-%d")
    ))
    plt.xticks(rotation=45)
    plt.tight_layout()

# 3.6 Compare label_interval distributions: flagged vs non-flagged
fig, ax = plt.subplots()
ax.hist(df.loc[~df.const_run_flag, "label_interval"],
        bins=50, alpha=0.5, density=True, label="ok rows")
ax.hist(df.loc[df.const_run_flag,  "label_interval"],
        bins=50, alpha=0.5, density=True, label="flagged rows")
ax.set_xlabel("call interval (min)")
ax.set_ylabel("density")
ax.set_title("Interval distribution – ok vs flagged")
ax.legend()
plt.tight_layout()

# 3.7 Tail heaviness metric before/after flag
def tail_ratio(s):
    return s.quantile(0.99) / s.quantile(0.95)

ratios = {
    "raw_all":    tail_ratio(df["label_interval"]),
    "ok_rows":    tail_ratio(df.loc[~df.const_run_flag, "label_interval"]),
    "flagged":    tail_ratio(df.loc[df.const_run_flag,  "label_interval"]),
}
fig, ax = plt.subplots()
ax.bar(ratios.keys(), ratios.values())
ax.set_ylabel("p99 / p95 ratio")
ax.set_title("Tail heaviness – raw vs flagged")
plt.xticks(rotation=15)
plt.tight_layout()

print("✅ Run the cells above to generate all requested diagnostics.")
```

**Usage notes:**

1. **Adjust** `SEQ_COLS`, `ORDER_COL`, `TEMP_COL`, and `CONST_RUN_MIN` if your pipeline differs.
2. This code computes:

   * **Row-level** and **segment-level** proportions flagged.
   * **Run-length** distributions and boxplots.
   * **Daily trend** of flagged rates.
   * **Interval distributions** for flagged vs. non-flagged rows.
   * **Tail-heaviness** ratios.
3. All plots are pure matplotlib and follow your charting guidelines. Copy/paste into your notebook and you’ll have rich visuals to prove the impact of the constant-run filter in your data.


# ────────────────────────────────────────────────────────────────────────────────
# Section 7 – Visual deep-dive on outliers & regime stability
# ────────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Helper: ensure tight, readable titles
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11})

----------------------------------


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
