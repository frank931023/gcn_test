import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from scipy import stats

# ── 設定 ──────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

COLORS = {"Normal": "#4C9BE8", "Blacklist": "#E84C4C"}

# ── 讀資料 & 清洗 ─────────────────────────────────────
user_info = pd.read_csv("data/user_info.csv")
train_label = pd.read_csv("data/train_label.csv")

TIME_COLS = ["confirmed_at", "level1_finished_at", "level2_finished_at"]
for col in TIME_COLS:
    user_info[col] = pd.to_datetime(user_info[col], errors="coerce")

user_info = user_info.dropna(subset=TIME_COLS, how="all")

df = user_info.merge(train_label[["user_id", "status"]], on="user_id", how="left")
df["status"] = df["status"].fillna(0).astype(int)
df["label"] = df["status"].map({0: "Normal", 1: "Blacklist"})

# ── 衍生速度特徵 ──────────────────────────────────────
# confirmed → level1 完成（秒）
df["l1_delay_sec"] = (df["level1_finished_at"] - df["confirmed_at"]).dt.total_seconds()
# confirmed → level2 完成（天）
df["l2_delay_days"] = (df["level2_finished_at"] - df["confirmed_at"]).dt.total_seconds() / 86400
# level1 → level2 間隔（天）
df["l1_to_l2_days"] = (df["level2_finished_at"] - df["level1_finished_at"]).dt.total_seconds() / 86400
# 認證小時（confirmed_at）
df["confirmed_hour"] = df["confirmed_at"].dt.hour
# 是否深夜認證（22:00 ~ 05:59）
df["is_night"] = df["confirmed_hour"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)

print(f"Total: {len(df)}  Blacklist: {df['status'].sum()}  Normal: {(df['status']==0).sum()}")

# ── 統計摘要 ──────────────────────────────────────────
for feat in ["l1_delay_sec", "l2_delay_days", "l1_to_l2_days"]:
    g0 = df[df["status"] == 0][feat].dropna()
    g1 = df[df["status"] == 1][feat].dropna()
    u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    print(f"\n[{feat}]")
    print(f"  Normal   median={g0.median():.2f}  mean={g0.mean():.2f}")
    print(f"  Blacklist median={g1.median():.2f}  mean={g1.mean():.2f}")
    print(f"  Mann-Whitney U p={p_val:.4e}")

# ═══════════════════════════════════════════════════════
# 圖 1：confirmed → level1 速度 Box Plot（去極端值）
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("KYC Speed: confirmed_at → level1_finished_at", fontsize=13)

valid_l1 = df[df["l1_delay_sec"].between(0, 3600)].copy()

# 左：整體 box
data_box = [valid_l1[valid_l1["status"] == 0]["l1_delay_sec"].dropna(),
            valid_l1[valid_l1["status"] == 1]["l1_delay_sec"].dropna()]
bp = axes[0].boxplot(data_box, patch_artist=True, notch=True,
                     medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticklabels(["Normal", "Blacklist"])
axes[0].set_ylabel("Seconds (0–3600)")
axes[0].set_title("Box Plot (within 1 hour)")

# 右：histogram 疊加
bins = np.linspace(0, 600, 50)
for label, grp in valid_l1[valid_l1["l1_delay_sec"] <= 600].groupby("label"):
    axes[1].hist(grp["l1_delay_sec"], bins=bins, alpha=0.6,
                 label=label, color=COLORS[label], density=True)
axes[1].set_xlabel("Seconds (0–600)")
axes[1].set_ylabel("Density")
axes[1].set_title("Distribution (within 10 min)")
axes[1].legend()

# 標注 p-value
g0 = valid_l1[valid_l1["status"] == 0]["l1_delay_sec"]
g1 = valid_l1[valid_l1["status"] == 1]["l1_delay_sec"]
_, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
axes[0].text(0.98, 0.97, f"p={p:.2e}", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_l1_speed_boxplot.png")
plt.close()
print("\nSaved 08_l1_speed_boxplot.png")

# ═══════════════════════════════════════════════════════
# 圖 2：confirmed → level2 速度 Box Plot
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("KYC Speed: confirmed_at → level2_finished_at", fontsize=13)

valid_l2 = df[df["l2_delay_days"].between(0, 1500)].copy()

data_box2 = [valid_l2[valid_l2["status"] == 0]["l2_delay_days"].dropna(),
             valid_l2[valid_l2["status"] == 1]["l2_delay_days"].dropna()]
bp2 = axes[0].boxplot(data_box2, patch_artist=True, notch=True,
                      medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp2["boxes"], COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticklabels(["Normal", "Blacklist"])
axes[0].set_ylabel("Days")
axes[0].set_title("Box Plot")

g0_l2 = valid_l2[valid_l2["status"] == 0]["l2_delay_days"]
g1_l2 = valid_l2[valid_l2["status"] == 1]["l2_delay_days"]
_, p_l2 = stats.mannwhitneyu(g0_l2, g1_l2, alternative="two-sided")
axes[0].text(0.98, 0.97, f"p={p_l2:.2e}", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

bins2 = np.linspace(0, 1500, 60)
for label, grp in valid_l2.groupby("label"):
    axes[1].hist(grp["l2_delay_days"], bins=bins2, alpha=0.6,
                 label=label, color=COLORS[label], density=True)
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Density")
axes[1].set_title("Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_l2_speed_boxplot.png")
plt.close()
print("Saved 09_l2_speed_boxplot.png")

# ═══════════════════════════════════════════════════════
# 圖 3：level1 → level2 間隔天數
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Gap: level1_finished_at → level2_finished_at", fontsize=13)

valid_gap = df[df["l1_to_l2_days"].between(0, 1500)].copy()

data_gap = [valid_gap[valid_gap["status"] == 0]["l1_to_l2_days"].dropna(),
            valid_gap[valid_gap["status"] == 1]["l1_to_l2_days"].dropna()]
bp3 = axes[0].boxplot(data_gap, patch_artist=True, notch=True,
                      medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp3["boxes"], COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticklabels(["Normal", "Blacklist"])
axes[0].set_ylabel("Days")
axes[0].set_title("Box Plot")

g0_gap = valid_gap[valid_gap["status"] == 0]["l1_to_l2_days"]
g1_gap = valid_gap[valid_gap["status"] == 1]["l1_to_l2_days"]
_, p_gap = stats.mannwhitneyu(g0_gap, g1_gap, alternative="two-sided")
axes[0].text(0.98, 0.97, f"p={p_gap:.2e}", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

bins3 = np.linspace(0, 1500, 60)
for label, grp in valid_gap.groupby("label"):
    axes[1].hist(grp["l1_to_l2_days"], bins=bins3, alpha=0.6,
                 label=label, color=COLORS[label], density=True)
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Density")
axes[1].set_title("Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_l1_to_l2_gap.png")
plt.close()
print("Saved 10_l1_to_l2_gap.png")

# ═══════════════════════════════════════════════════════
# 圖 4：認證小時分佈（Normal vs Blacklist）
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("confirmed_at Hour-of-Day: Normal vs Blacklist (normalized)", fontsize=13)

x = np.arange(24)
width = 0.4
for i, (label, grp) in enumerate(df.groupby("label")):
    counts = grp["confirmed_hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
    norm = counts / counts.sum() * 100
    ax.bar(x + i * width, norm, width, label=label,
           color=COLORS[label], alpha=0.8)

ax.set_xticks(x + width / 2)
ax.set_xticklabels([str(h) for h in range(24)])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("% of users")
ax.axvspan(-0.5, 5.5 + width, alpha=0.07, color="navy", label="Night (22–05)")
ax.axvspan(21.5, 23.5 + width, alpha=0.07, color="navy")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_confirmed_hour_dist.png")
plt.close()
print("Saved 11_confirmed_hour_dist.png")

# ═══════════════════════════════════════════════════════
# 圖 5：深夜認證比例（Normal vs Blacklist）
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Night-time Registration Rate (22:00–05:59)", fontsize=13)

night_rate = df.groupby("label")["is_night"].mean() * 100
bars = ax.bar(night_rate.index, night_rate.values,
              color=[COLORS[l] for l in night_rate.index], alpha=0.8, width=0.4)
for bar, val in zip(bars, night_rate.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

# chi-square test
ct = pd.crosstab(df["status"], df["is_night"])
chi2, p_chi, _, _ = stats.chi2_contingency(ct)
ax.text(0.98, 0.97, f"χ² p={p_chi:.2e}", transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

ax.set_ylabel("% of users")
ax.set_ylim(0, max(night_rate.values) * 1.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_night_registration_rate.png")
plt.close()
print("Saved 12_night_registration_rate.png")

# ═══════════════════════════════════════════════════════
# 圖 6：速度特徵 median 比較總覽
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Median Comparison: Normal vs Blacklist", fontsize=13)

features = [
    ("l1_delay_sec",   "confirmed→level1 (sec)",  df[df["l1_delay_sec"].between(0, 3600)]),
    ("l2_delay_days",  "confirmed→level2 (days)",  df[df["l2_delay_days"].between(0, 1500)]),
    ("l1_to_l2_days",  "level1→level2 (days)",     df[df["l1_to_l2_days"].between(0, 1500)]),
]

for ax, (feat, title, sub) in zip(axes, features):
    medians = sub.groupby("label")[feat].median()
    bars = ax.bar(medians.index, medians.values,
                  color=[COLORS[l] for l in medians.index], alpha=0.8, width=0.4)
    for bar, val in zip(bars, medians.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Median")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_speed_median_comparison.png")
plt.close()
print("Saved 13_speed_median_comparison.png")

print("\nAll done. Outputs saved to outputs/")
