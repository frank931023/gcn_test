"""
閃電過水時間差視覺化
比較黑名單 (status=1) 與正常用戶 (status=0) 在以下行為的差異：
1. KYC Level2 完成 → 首次 TWD 入金 的時間差 (ΔT_deposit)
2. 首次 TWD 入金 → 首次 Crypto 出金 的時間差 (ΔT_crypto)
3. 入金後快速出金的行為模式
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── 字型設定 ─────────────────────────────────────────────────────
plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_NORMAL = "#4C9BE8"
COLOR_BLACK  = "#E84C4C"
ALPHA = 0.7

# ── 讀取資料 ─────────────────────────────────────────────────────
print("Loading data...")
user_info       = pd.read_csv("data/user_info.csv",      parse_dates=["level2_finished_at"])
twd_transfer    = pd.read_csv("data/twd_transfer.csv",   parse_dates=["created_at"])
crypto_transfer = pd.read_csv("data/crypto_transfer.csv",parse_dates=["created_at"])
labels          = pd.read_csv("data/train_label.csv")

user_info = user_info.merge(labels, on="user_id", how="left")
user_info["status"] = user_info["status"].fillna(0).astype(int)

# ── 特徵 1：KYC Level2 → 首次 TWD 入金 ──────────────────────────
twd_deposit  = twd_transfer[twd_transfer["kind"] == 0]
first_deposit = (
    twd_deposit.sort_values("created_at")
    .groupby("user_id")["created_at"].first()
    .reset_index().rename(columns={"created_at": "first_deposit_at"})
)

df = user_info[["user_id", "level2_finished_at", "status"]].merge(
    first_deposit, on="user_id", how="inner"
)
df["delta_kyc_to_deposit_days"] = (
    df["first_deposit_at"] - df["level2_finished_at"]
).dt.total_seconds() / 86400
df = df[df["delta_kyc_to_deposit_days"] >= 0]

# ── 特徵 2：首次 TWD 入金 → 首次 Crypto 出金 ────────────────────
crypto_withdraw = crypto_transfer[crypto_transfer["kind"] == 1]
first_crypto = (
    crypto_withdraw.sort_values("created_at")
    .groupby("user_id")["created_at"].first()
    .reset_index().rename(columns={"created_at": "first_crypto_at"})
)

df = df.merge(first_crypto, on="user_id", how="left")
df["delta_deposit_to_crypto_h"] = (
    df["first_crypto_at"] - df["first_deposit_at"]
).dt.total_seconds() / 3600
df["has_crypto_withdraw"] = df["delta_deposit_to_crypto_h"].notna()

# ── 特徵 3：入金後 72h 內出金（過水行為）────────────────────────
df["is_flash"] = (
    df["delta_deposit_to_crypto_h"].notna() &
    (df["delta_deposit_to_crypto_h"] >= 0) &
    (df["delta_deposit_to_crypto_h"] < 72)
)

# ── 特徵 4：KYC 後 30 天內入金 ──────────────────────────────────
df["fast_deposit_30d"] = df["delta_kyc_to_deposit_days"] < 30

normal = df[df["status"] == 0]
black  = df[df["status"] == 1]

print(f"Normal users: {len(normal):,}  Blacklist users: {len(black):,}")
print(f"Normal - median KYC->deposit: {normal['delta_kyc_to_deposit_days'].median():.0f} days")
print(f"Black  - median KYC->deposit: {black['delta_kyc_to_deposit_days'].median():.0f} days")
print(f"Normal - flash withdraw rate: {normal['is_flash'].mean()*100:.1f}%")
print(f"Black  - flash withdraw rate: {black['is_flash'].mean()*100:.1f}%")
print(f"Normal - has crypto withdraw: {normal['has_crypto_withdraw'].mean()*100:.1f}%")
print(f"Black  - has crypto withdraw: {black['has_crypto_withdraw'].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════
# 繪圖
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 15))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle("Flash Throughput Velocity Analysis\nKYC Level2 -> First TWD Deposit -> First Crypto Withdrawal",
             fontsize=15, fontweight="bold", y=0.99)

gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

# ── 圖1：KYC→入金 天數分佈（log scale）──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor("#FAFAFA")
bins = np.logspace(np.log10(1), np.log10(3650), 60)
ax1.hist(normal["delta_kyc_to_deposit_days"].clip(1, 3650), bins=bins,
         color=COLOR_NORMAL, alpha=ALPHA, label=f"Normal (n={len(normal):,})", density=True)
ax1.hist(black["delta_kyc_to_deposit_days"].clip(1, 3650), bins=bins,
         color=COLOR_BLACK, alpha=ALPHA, label=f"Blacklist (n={len(black):,})", density=True)
ax1.axvline(30, color="orange", linestyle="--", linewidth=2, label="30-day threshold")
ax1.axvline(7,  color="red",    linestyle=":",  linewidth=1.5, label="7-day threshold")
ax1.set_xscale("log")
ax1.set_xlabel("KYC Level2 -> First Deposit (days, log scale)")
ax1.set_ylabel("Density")
ax1.set_title("(1) KYC to First Deposit Time Distribution")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# ── 圖2：KYC 後 30 天內入金比例 ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor("#FAFAFA")
groups = ["Normal", "Blacklist"]
pct_30d = [
    normal["fast_deposit_30d"].mean() * 100,
    black["fast_deposit_30d"].mean() * 100,
]
bars = ax2.bar(groups, pct_30d, color=[COLOR_NORMAL, COLOR_BLACK], alpha=ALPHA, width=0.5)
for bar, val in zip(bars, pct_30d):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax2.set_ylabel("Percentage (%)")
ax2.set_title("(2) Deposit within 30 days\nafter KYC")
ax2.set_ylim(0, max(pct_30d) * 1.35 + 1)
ax2.grid(axis="y", alpha=0.3)

# ── 圖3：入金→Crypto出金 小時分佈 ───────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor("#FAFAFA")
df_c = df[df["delta_deposit_to_crypto_h"].notna() & (df["delta_deposit_to_crypto_h"] >= 0)]
n3 = df_c[df_c["status"] == 0]
b3 = df_c[df_c["status"] == 1]
bins3 = np.logspace(np.log10(0.1), np.log10(8760), 60)
ax3.hist(n3["delta_deposit_to_crypto_h"].clip(0.1, 8760), bins=bins3,
         color=COLOR_NORMAL, alpha=ALPHA, label=f"Normal (n={len(n3):,})", density=True)
ax3.hist(b3["delta_deposit_to_crypto_h"].clip(0.1, 8760), bins=bins3,
         color=COLOR_BLACK, alpha=ALPHA, label=f"Blacklist (n={len(b3):,})", density=True)
ax3.axvline(72, color="orange", linestyle="--", linewidth=2, label="72h threshold")
ax3.axvline(24, color="red",    linestyle=":",  linewidth=1.5, label="24h threshold")
ax3.set_xscale("log")
ax3.set_xlabel("First Deposit -> First Crypto Withdrawal (hours, log scale)")
ax3.set_ylabel("Density")
ax3.set_title("(3) Deposit to Crypto Withdrawal Time Distribution")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

# ── 圖4：過水比例 & Crypto出金比例 ──────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor("#FAFAFA")
metrics = ["Flash\n(<72h)", "Has Crypto\nWithdraw"]
normal_vals = [normal["is_flash"].mean()*100, normal["has_crypto_withdraw"].mean()*100]
black_vals  = [black["is_flash"].mean()*100,  black["has_crypto_withdraw"].mean()*100]
x = np.arange(len(metrics))
w = 0.35
b1 = ax4.bar(x - w/2, normal_vals, w, color=COLOR_NORMAL, alpha=ALPHA, label="Normal")
b2 = ax4.bar(x + w/2, black_vals,  w, color=COLOR_BLACK,  alpha=ALPHA, label="Blacklist")
for bar, val in zip(list(b1)+list(b2), normal_vals+black_vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax4.set_xticks(x); ax4.set_xticklabels(metrics)
ax4.set_ylabel("Percentage (%)")
ax4.set_title("(4) Flash Withdraw &\nCrypto Activity Rate")
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.3)

# ── 圖5：散點圖 ΔT1 vs ΔT2 ──────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
ax5.set_facecolor("#FAFAFA")
scatter_df = df[
    df["delta_deposit_to_crypto_h"].notna() &
    (df["delta_deposit_to_crypto_h"] > 0) &
    (df["delta_kyc_to_deposit_days"] > 0)
].copy()
sn = scatter_df[scatter_df["status"] == 0].sample(
    min(800, len(scatter_df[scatter_df["status"]==0])), random_state=42)
sb = scatter_df[scatter_df["status"] == 1]

ax5.scatter(sn["delta_kyc_to_deposit_days"], sn["delta_deposit_to_crypto_h"],
            c=COLOR_NORMAL, alpha=0.35, s=12, label=f"Normal (sample n={len(sn)})")
ax5.scatter(sb["delta_kyc_to_deposit_days"], sb["delta_deposit_to_crypto_h"],
            c=COLOR_BLACK, alpha=0.65, s=18, label=f"Blacklist (n={len(sb)})")
ax5.axvline(30, color="orange", linestyle="--", linewidth=1.2, alpha=0.8, label="30d")
ax5.axhline(72, color="purple", linestyle="--", linewidth=1.2, alpha=0.8, label="72h")
ax5.set_xscale("log"); ax5.set_yscale("log")
ax5.set_xlabel("KYC -> First Deposit (days)")
ax5.set_ylabel("First Deposit -> Crypto Withdrawal (hours)")
ax5.set_title("(5) Dual Time-Gap Scatter (bottom-left = high-risk zone)")
ax5.legend(markerscale=2, fontsize=9)
ax5.grid(alpha=0.2)
# 高風險區標記
ax5.add_patch(mpatches.Rectangle(
    (0.5, 0.1), 30, 72,
    linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.06,
    transform=ax5.transData))
ax5.text(0.7, 0.15, "High-Risk\nZone", color="red", fontsize=9, alpha=0.75)

# ── 圖6：KYC→入金 CDF 比較 ──────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor("#FAFAFA")
for grp, color, label in [(normal, COLOR_NORMAL, "Normal"), (black, COLOR_BLACK, "Blacklist")]:
    vals = grp["delta_kyc_to_deposit_days"].dropna().sort_values()
    cdf  = np.arange(1, len(vals)+1) / len(vals)
    ax6.plot(vals, cdf, color=color, label=label, linewidth=2.5)
ax6.axvline(30,  color="orange", linestyle="--", linewidth=1.5, label="30d")
ax6.axvline(7,   color="red",    linestyle=":",  linewidth=1.2, label="7d")
ax6.axvline(365, color="gray",   linestyle="-.", linewidth=1,   label="1yr")
ax6.set_xscale("log")
ax6.set_xlabel("KYC -> First Deposit (days)")
ax6.set_ylabel("Cumulative Proportion (CDF)")
ax6.set_title("(6) KYC->Deposit CDF Comparison")
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

plt.savefig("time/flash_throughput_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nSaved: time/flash_throughput_analysis.png")

# ── 統計摘要 ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("STATISTICAL SUMMARY - Flash Throughput Velocity")
print("="*65)
thresholds = [7, 14, 30, 90, 180, 365]
print(f"\n{'Threshold':<15} {'Normal %':>10} {'Blacklist %':>12} {'Ratio':>8}")
print("-"*50)
for t in thresholds:
    n_pct = (normal["delta_kyc_to_deposit_days"] < t).mean() * 100
    b_pct = (black["delta_kyc_to_deposit_days"] < t).mean() * 100
    ratio = b_pct / n_pct if n_pct > 0 else float("inf")
    print(f"KYC < {t:>3}d     {n_pct:>9.1f}%  {b_pct:>10.1f}%  {ratio:>7.2f}x")

print(f"\nMedian KYC->Deposit:")
print(f"  Normal:    {normal['delta_kyc_to_deposit_days'].median():.0f} days")
print(f"  Blacklist: {black['delta_kyc_to_deposit_days'].median():.0f} days")
print(f"\nFlash Withdraw Rate (deposit->crypto < 72h):")
print(f"  Normal:    {normal['is_flash'].mean()*100:.2f}%")
print(f"  Blacklist: {black['is_flash'].mean()*100:.2f}%")
print(f"\nHas Any Crypto Withdrawal:")
print(f"  Normal:    {normal['has_crypto_withdraw'].mean()*100:.1f}%")
print(f"  Blacklist: {black['has_crypto_withdraw'].mean()*100:.1f}%")
