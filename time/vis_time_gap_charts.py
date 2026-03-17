"""
黑白名單時間差統計圖表
- ΔT1: KYC Level2 完成 → 首次 TWD 入金（天）
- ΔT2: 首次 TWD 入金 → 首次 Crypto 出金（小時）
每張圖獨立輸出
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "time/outputs_time_gap"
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_N = "#4C9BE8"   # 正常
COLOR_B = "#E84C4C"   # 黑名單
ALPHA   = 0.72

# ── 載入資料 ─────────────────────────────────────────────────────
print("Loading data...")
user_info       = pd.read_csv("data/user_info.csv",       parse_dates=["level2_finished_at"])
twd_transfer    = pd.read_csv("data/twd_transfer.csv",    parse_dates=["created_at"])
crypto_transfer = pd.read_csv("data/crypto_transfer.csv", parse_dates=["created_at"])
labels          = pd.read_csv("data/train_label.csv")

user_info = user_info.merge(labels, on="user_id", how="left")
user_info["status"] = user_info["status"].fillna(0).astype(int)

# ── 計算 ΔT1 ─────────────────────────────────────────────────────
first_deposit = (
    twd_transfer[twd_transfer["kind"] == 0]
    .sort_values("created_at")
    .groupby("user_id")["created_at"].first()
    .reset_index().rename(columns={"created_at": "first_deposit_at"})
)
df = user_info[["user_id", "level2_finished_at", "status"]].merge(
    first_deposit, on="user_id", how="inner"
)
df["dt1_days"] = (df["first_deposit_at"] - df["level2_finished_at"]).dt.total_seconds() / 86400
df = df[df["dt1_days"] >= 0]

# ── 計算 ΔT2 ─────────────────────────────────────────────────────
first_crypto = (
    crypto_transfer[crypto_transfer["kind"] == 1]
    .sort_values("created_at")
    .groupby("user_id")["created_at"].first()
    .reset_index().rename(columns={"created_at": "first_crypto_at"})
)
df = df.merge(first_crypto, on="user_id", how="left")
df["dt2_hours"] = (df["first_crypto_at"] - df["first_deposit_at"]).dt.total_seconds() / 3600
df2 = df[df["dt2_hours"].notna() & (df["dt2_hours"] >= 0)].copy()

# ── 去除極端值（IQR 方法，各組分別計算）────────────────────────
def remove_outliers_iqr(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series[(series >= q1 - k*iqr) & (series <= q3 + k*iqr)]

normal_dt1 = remove_outliers_iqr(df[df["status"]==0]["dt1_days"])
black_dt1  = remove_outliers_iqr(df[df["status"]==1]["dt1_days"])
normal_dt2 = remove_outliers_iqr(df2[df2["status"]==0]["dt2_hours"])
black_dt2  = remove_outliers_iqr(df2[df2["status"]==1]["dt2_hours"])

print(f"ΔT1 - Normal: {len(normal_dt1):,}  Black: {len(black_dt1):,}")
print(f"ΔT2 - Normal: {len(normal_dt2):,}  Black: {len(black_dt2):,}")

# ════════════════════════════════════════════════════════════════
# 共用函式
# ════════════════════════════════════════════════════════════════
def save(fig, name):
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")

def stat_text(series, label):
    return (f"{label}\n"
            f"n={len(series):,}\n"
            f"median={series.median():.1f}\n"
            f"mean={series.mean():.1f}\n"
            f"std={series.std():.1f}\n"
            f"p25={series.quantile(0.25):.1f}\n"
            f"p75={series.quantile(0.75):.1f}")

# ════════════════════════════════════════════════════════════════
# 圖 1：ΔT1 Histogram（重疊）
# ════════════════════════════════════════════════════════════════
print("\n[1] ΔT1 Histogram")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
bins = np.linspace(0, max(normal_dt1.max(), black_dt1.max()), 60)
ax.hist(normal_dt1, bins=bins, color=COLOR_N, alpha=ALPHA,
        label=f"Normal (n={len(normal_dt1):,})", density=True)
ax.hist(black_dt1,  bins=bins, color=COLOR_B, alpha=ALPHA,
        label=f"Blacklist (n={len(black_dt1):,})", density=True)
ax.axvline(normal_dt1.median(), color=COLOR_N, linestyle="--", linewidth=1.8,
           label=f"Normal median={normal_dt1.median():.0f}d")
ax.axvline(black_dt1.median(),  color=COLOR_B, linestyle="--", linewidth=1.8,
           label=f"Black median={black_dt1.median():.0f}d")
ax.set_xlabel("KYC Level2 -> First Deposit (days)")
ax.set_ylabel("Density")
ax.set_title("ΔT1: KYC Level2 Completion to First TWD Deposit\n(Histogram, outliers removed by IQR×3)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "01_dt1_histogram")

# ════════════════════════════════════════════════════════════════
# 圖 2：ΔT1 Box Plot
# ════════════════════════════════════════════════════════════════
print("[2] ΔT1 Box Plot")
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
bp = ax.boxplot([normal_dt1, black_dt1],
                labels=["Normal", "Blacklist"],
                patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
bp["boxes"][0].set_facecolor(COLOR_N); bp["boxes"][0].set_alpha(ALPHA)
bp["boxes"][1].set_facecolor(COLOR_B); bp["boxes"][1].set_alpha(ALPHA)
for i, (data, color) in enumerate([(normal_dt1, COLOR_N), (black_dt1, COLOR_B)], 1):
    ax.text(i, data.median() + 2, f"median\n{data.median():.0f}d",
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
ax.set_ylabel("Days")
ax.set_title("ΔT1: KYC Level2 -> First Deposit\n(Box Plot with Notch)")
ax.grid(axis="y", alpha=0.3)
save(fig, "02_dt1_boxplot")

# ════════════════════════════════════════════════════════════════
# 圖 3：ΔT1 Violin Plot
# ════════════════════════════════════════════════════════════════
print("[3] ΔT1 Violin Plot")
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
parts = ax.violinplot([normal_dt1, black_dt1], positions=[1, 2],
                      showmedians=True, showextrema=True)
colors = [COLOR_N, COLOR_B]
for i, (pc, c) in enumerate(zip(parts["bodies"], colors)):
    pc.set_facecolor(c); pc.set_alpha(0.65)
parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
parts["cmaxes"].set_linewidth(1.5); parts["cmins"].set_linewidth(1.5)
parts["cbars"].set_linewidth(1.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Blacklist"])
ax.set_ylabel("Days")
ax.set_title("ΔT1: KYC Level2 -> First Deposit\n(Violin Plot)")
ax.grid(axis="y", alpha=0.3)
# 加上 median 標籤
for pos, data, color in [(1, normal_dt1, COLOR_N), (2, black_dt1, COLOR_B)]:
    ax.text(pos, data.median() + 2, f"{data.median():.0f}d",
            ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")
save(fig, "03_dt1_violin")

# ════════════════════════════════════════════════════════════════
# 圖 4：ΔT1 CDF
# ════════════════════════════════════════════════════════════════
print("[4] ΔT1 CDF")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
for data, color, label in [(normal_dt1, COLOR_N, "Normal"), (black_dt1, COLOR_B, "Blacklist")]:
    s = data.sort_values()
    cdf = np.arange(1, len(s)+1) / len(s)
    ax.plot(s, cdf, color=color, linewidth=2.5, label=label)
for thresh, ls, label in [(7,":",  "7d"), (30,"--","30d"), (90,"-.","90d")]:
    ax.axvline(thresh, color="gray", linestyle=ls, linewidth=1.2, alpha=0.7, label=label)
ax.set_xlabel("KYC Level2 -> First Deposit (days)")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("ΔT1: KYC Level2 -> First Deposit\n(CDF Comparison)")
ax.legend(); ax.grid(alpha=0.25)
# 標記各門檻的差距
for thresh in [7, 30, 90]:
    n_pct = (normal_dt1 <= thresh).mean()
    b_pct = (black_dt1  <= thresh).mean()
    ax.annotate(f"@{thresh}d\nN:{n_pct*100:.0f}%\nB:{b_pct*100:.0f}%",
                xy=(thresh, (n_pct+b_pct)/2),
                xytext=(thresh+8, (n_pct+b_pct)/2),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
save(fig, "04_dt1_cdf")

# ════════════════════════════════════════════════════════════════
# 圖 5：ΔT1 統計摘要表（橫條圖）
# ════════════════════════════════════════════════════════════════
print("[5] ΔT1 Stats Bar")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
stat_names = ["Mean", "Median", "P25", "P75", "Std"]
n_vals = [normal_dt1.mean(), normal_dt1.median(),
          normal_dt1.quantile(0.25), normal_dt1.quantile(0.75), normal_dt1.std()]
b_vals = [black_dt1.mean(),  black_dt1.median(),
          black_dt1.quantile(0.25),  black_dt1.quantile(0.75),  black_dt1.std()]
x = np.arange(len(stat_names))
w = 0.35
bars_n = ax.bar(x - w/2, n_vals, w, color=COLOR_N, alpha=ALPHA, label="Normal")
bars_b = ax.bar(x + w/2, b_vals, w, color=COLOR_B, alpha=ALPHA, label="Blacklist")
for bar, val in zip(list(bars_n)+list(bars_b), n_vals+b_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(stat_names)
ax.set_ylabel("Days")
ax.set_title("ΔT1: KYC Level2 -> First Deposit\n(Descriptive Statistics Comparison)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "05_dt1_stats_bar")

# ════════════════════════════════════════════════════════════════
# 圖 6：ΔT2 Histogram
# ════════════════════════════════════════════════════════════════
print("[6] ΔT2 Histogram")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
bins2 = np.linspace(0, max(normal_dt2.max(), black_dt2.max()), 60)
ax.hist(normal_dt2, bins=bins2, color=COLOR_N, alpha=ALPHA,
        label=f"Normal (n={len(normal_dt2):,})", density=True)
ax.hist(black_dt2,  bins=bins2, color=COLOR_B, alpha=ALPHA,
        label=f"Blacklist (n={len(black_dt2):,})", density=True)
ax.axvline(normal_dt2.median(), color=COLOR_N, linestyle="--", linewidth=1.8,
           label=f"Normal median={normal_dt2.median():.1f}h")
ax.axvline(black_dt2.median(),  color=COLOR_B, linestyle="--", linewidth=1.8,
           label=f"Black median={black_dt2.median():.1f}h")
ax.set_xlabel("First Deposit -> First Crypto Withdrawal (hours)")
ax.set_ylabel("Density")
ax.set_title("ΔT2: First TWD Deposit to First Crypto Withdrawal\n(Histogram, outliers removed by IQR×3)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "06_dt2_histogram")

# ════════════════════════════════════════════════════════════════
# 圖 7：ΔT2 Box Plot
# ════════════════════════════════════════════════════════════════
print("[7] ΔT2 Box Plot")
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
bp = ax.boxplot([normal_dt2, black_dt2],
                labels=["Normal", "Blacklist"],
                patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
bp["boxes"][0].set_facecolor(COLOR_N); bp["boxes"][0].set_alpha(ALPHA)
bp["boxes"][1].set_facecolor(COLOR_B); bp["boxes"][1].set_alpha(ALPHA)
for i, (data, color) in enumerate([(normal_dt2, COLOR_N), (black_dt2, COLOR_B)], 1):
    ax.text(i, data.median() + 0.5, f"median\n{data.median():.1f}h",
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
ax.set_ylabel("Hours")
ax.set_title("ΔT2: First Deposit -> First Crypto Withdrawal\n(Box Plot with Notch)")
ax.grid(axis="y", alpha=0.3)
save(fig, "07_dt2_boxplot")

# ════════════════════════════════════════════════════════════════
# 圖 8：ΔT2 Violin Plot
# ════════════════════════════════════════════════════════════════
print("[8] ΔT2 Violin Plot")
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
parts = ax.violinplot([normal_dt2, black_dt2], positions=[1, 2],
                      showmedians=True, showextrema=True)
for pc, c in zip(parts["bodies"], [COLOR_N, COLOR_B]):
    pc.set_facecolor(c); pc.set_alpha(0.65)
parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
parts["cmaxes"].set_linewidth(1.5); parts["cmins"].set_linewidth(1.5)
parts["cbars"].set_linewidth(1.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Blacklist"])
ax.set_ylabel("Hours")
ax.set_title("ΔT2: First Deposit -> First Crypto Withdrawal\n(Violin Plot)")
ax.grid(axis="y", alpha=0.3)
for pos, data, color in [(1, normal_dt2, COLOR_N), (2, black_dt2, COLOR_B)]:
    ax.text(pos, data.median() + 0.3, f"{data.median():.1f}h",
            ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")
save(fig, "08_dt2_violin")

# ════════════════════════════════════════════════════════════════
# 圖 9：ΔT2 CDF
# ════════════════════════════════════════════════════════════════
print("[9] ΔT2 CDF")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
for data, color, label in [(normal_dt2, COLOR_N, "Normal"), (black_dt2, COLOR_B, "Blacklist")]:
    s = data.sort_values()
    cdf = np.arange(1, len(s)+1) / len(s)
    ax.plot(s, cdf, color=color, linewidth=2.5, label=label)
for thresh, ls, label in [(24,":", "24h"), (72,"--","72h"), (168,"-.","1wk")]:
    ax.axvline(thresh, color="gray", linestyle=ls, linewidth=1.2, alpha=0.7, label=label)
ax.set_xlabel("First Deposit -> First Crypto Withdrawal (hours)")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("ΔT2: First Deposit -> First Crypto Withdrawal\n(CDF Comparison)")
ax.legend(); ax.grid(alpha=0.25)
for thresh in [24, 72]:
    n_pct = (normal_dt2 <= thresh).mean()
    b_pct = (black_dt2  <= thresh).mean()
    ax.annotate(f"@{thresh}h\nN:{n_pct*100:.0f}%\nB:{b_pct*100:.0f}%",
                xy=(thresh, (n_pct+b_pct)/2),
                xytext=(thresh+10, (n_pct+b_pct)/2 - 0.05),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
save(fig, "09_dt2_cdf")

# ════════════════════════════════════════════════════════════════
# 圖 10：ΔT2 統計摘要表
# ════════════════════════════════════════════════════════════════
print("[10] ΔT2 Stats Bar")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
n_vals2 = [normal_dt2.mean(), normal_dt2.median(),
           normal_dt2.quantile(0.25), normal_dt2.quantile(0.75), normal_dt2.std()]
b_vals2 = [black_dt2.mean(),  black_dt2.median(),
           black_dt2.quantile(0.25),  black_dt2.quantile(0.75),  black_dt2.std()]
x = np.arange(len(stat_names))
bars_n = ax.bar(x - w/2, n_vals2, w, color=COLOR_N, alpha=ALPHA, label="Normal")
bars_b = ax.bar(x + w/2, b_vals2, w, color=COLOR_B, alpha=ALPHA, label="Blacklist")
for bar, val in zip(list(bars_n)+list(bars_b), n_vals2+b_vals2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(stat_names)
ax.set_ylabel("Hours")
ax.set_title("ΔT2: First Deposit -> First Crypto Withdrawal\n(Descriptive Statistics Comparison)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "10_dt2_stats_bar")

# ════════════════════════════════════════════════════════════════
# 圖 11：ΔT1 散點圖（jitter strip plot）
# ════════════════════════════════════════════════════════════════
print("[11] ΔT1 Strip Plot")
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
rng = np.random.default_rng(42)
for pos, data, color, label in [
    (1, normal_dt1.sample(min(600, len(normal_dt1)), random_state=42), COLOR_N, "Normal"),
    (2, black_dt1,  COLOR_B, "Blacklist")
]:
    jitter = rng.uniform(-0.18, 0.18, len(data))
    ax.scatter(np.full(len(data), pos) + jitter, data,
               color=color, alpha=0.35, s=10, label=f"{label} (n={len(data):,})")
    ax.hlines(data.median(), pos-0.3, pos+0.3, color=color, linewidth=2.5,
              label=f"{label} median={data.median():.0f}d")
ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Blacklist"])
ax.set_ylabel("Days")
ax.set_title("ΔT1: KYC Level2 -> First Deposit\n(Strip Plot with Median Line)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
save(fig, "11_dt1_strip")

# ════════════════════════════════════════════════════════════════
# 圖 12：ΔT2 散點圖（jitter strip plot）
# ════════════════════════════════════════════════════════════════
print("[12] ΔT2 Strip Plot")
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
for pos, data, color, label in [
    (1, normal_dt2.sample(min(600, len(normal_dt2)), random_state=42), COLOR_N, "Normal"),
    (2, black_dt2,  COLOR_B, "Blacklist")
]:
    jitter = rng.uniform(-0.18, 0.18, len(data))
    ax.scatter(np.full(len(data), pos) + jitter, data,
               color=color, alpha=0.35, s=10, label=f"{label} (n={len(data):,})")
    ax.hlines(data.median(), pos-0.3, pos+0.3, color=color, linewidth=2.5,
              label=f"{label} median={data.median():.1f}h")
ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Blacklist"])
ax.set_ylabel("Hours")
ax.set_title("ΔT2: First Deposit -> First Crypto Withdrawal\n(Strip Plot with Median Line)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
save(fig, "12_dt2_strip")

# ════════════════════════════════════════════════════════════════
# 統計摘要輸出
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SUMMARY TABLE")
print("="*65)
print(f"\n{'Metric':<20} {'Normal ΔT1':>12} {'Black ΔT1':>12} {'Normal ΔT2':>12} {'Black ΔT2':>12}")
print("-"*70)
rows = [
    ("n",      len(normal_dt1), len(black_dt1), len(normal_dt2), len(black_dt2)),
    ("mean",   normal_dt1.mean(), black_dt1.mean(), normal_dt2.mean(), black_dt2.mean()),
    ("median", normal_dt1.median(), black_dt1.median(), normal_dt2.median(), black_dt2.median()),
    ("std",    normal_dt1.std(), black_dt1.std(), normal_dt2.std(), black_dt2.std()),
    ("p25",    normal_dt1.quantile(.25), black_dt1.quantile(.25), normal_dt2.quantile(.25), black_dt2.quantile(.25)),
    ("p75",    normal_dt1.quantile(.75), black_dt1.quantile(.75), normal_dt2.quantile(.75), black_dt2.quantile(.75)),
]
for row in rows:
    print(f"{row[0]:<20} {row[1]:>12.1f} {row[2]:>12.1f} {row[3]:>12.1f} {row[4]:>12.1f}")

# Mann-Whitney U test
u1, p1 = stats.mannwhitneyu(normal_dt1, black_dt1, alternative="two-sided")
u2, p2 = stats.mannwhitneyu(normal_dt2, black_dt2, alternative="two-sided")
print(f"\nMann-Whitney U test:")
print(f"  ΔT1: U={u1:.0f}, p={p1:.2e}  {'***' if p1<0.001 else '**' if p1<0.01 else '*' if p1<0.05 else 'ns'}")
print(f"  ΔT2: U={u2:.0f}, p={p2:.2e}  {'***' if p2<0.001 else '**' if p2<0.01 else '*' if p2<0.05 else 'ns'}")
print(f"\nAll charts saved to: {OUT_DIR}/")
