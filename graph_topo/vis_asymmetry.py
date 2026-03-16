"""
驗證黑名單帳戶是否具有「極端不對稱度數」
out_degree >> in_degree（大量匯出）或 in_degree >> out_degree（大量匯入/集資）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "graph_topo", "outputs")
os.makedirs(OUT, exist_ok=True)

# ── 載入 ──────────────────────────────────────────────────
crypto = pd.read_csv(os.path.join(DATA, "crypto_transfer.csv"))
train  = pd.read_csv(os.path.join(DATA, "train_label.csv"))

out_deg = crypto.groupby("user_id").size().rename("out_degree")
in_deg  = (crypto[crypto["relation_user_id"].notna()]
           .groupby("relation_user_id").size().rename("in_degree"))

df = (train.join(out_deg, on="user_id").join(in_deg, on="user_id").fillna(0))
df["out_degree"] = df["out_degree"].astype(int)
df["in_degree"]  = df["in_degree"].astype(int)
df["total"]      = df["out_degree"] + df["in_degree"]

# 不對稱比：(out - in) / (out + in)，範圍 [-1, 1]
# +1 = 純匯出（水房）, -1 = 純匯入（集資錢包）, 0 = 對稱
df["asym"] = np.where(
    df["total"] > 0,
    (df["out_degree"] - df["in_degree"]) / df["total"],
    0.0
)

black  = df[df["status"] == 1]
normal = df[df["status"] == 0]

print("=== 不對稱比 (asym) 統計 ===")
for grp, name in [(black, "黑名單"), (normal, "正常")]:
    print(f"\n{name}:")
    print(grp["asym"].describe().round(3))
    print(f"  |asym| > 0.8 的比例: {(grp['asym'].abs() > 0.8).mean():.1%}")

# ── 圖 1：不對稱比分布（KDE 密度）────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(-1, 1, 60)
ax.hist(normal["asym"], bins=bins, color="seagreen", alpha=0.5,
        density=True, label="Normal (0)")
ax.hist(black["asym"],  bins=bins, color="crimson",  alpha=0.6,
        density=True, label="Blacklist (1)")
ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Asymmetry  =  (out − in) / (out + in)", fontsize=11)
ax.set_ylabel("Density")
ax.set_title("Degree Asymmetry Distribution\n+1=pure sender, -1=pure receiver", fontsize=12)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "asym_hist.png"), dpi=150)
plt.close()
print("Saved: asym_hist.png")

# ── 圖 2：散佈圖（out vs in），顏色=label，大小=|asym|）──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cap_out = int(df["out_degree"].quantile(0.97))
cap_in  = int(df["in_degree"].quantile(0.97))

for ax, grp, title, color in zip(
    axes,
    [normal, black],
    ["Normal (0)", "Blacklist (1)"],
    ["seagreen", "crimson"]
):
    sizes = (grp["asym"].abs() * 40 + 5).clip(upper=80)
    ax.scatter(
        grp["out_degree"].clip(upper=cap_out),
        grp["in_degree"].clip(upper=cap_in),
        c=color, alpha=0.4, s=sizes, edgecolors="none"
    )
    # 標出極端不對稱點（|asym| > 0.9 且 total > 5）
    extreme = grp[(grp["asym"].abs() > 0.9) & (grp["total"] > 5)]
    ax.scatter(
        extreme["out_degree"].clip(upper=cap_out),
        extreme["in_degree"].clip(upper=cap_in),
        c="gold", s=60, edgecolors="black", linewidths=0.6,
        zorder=5, label=f"Extreme |asym|>0.9 (n={len(extreme)})"
    )
    ax.set_xlabel("Out-Degree")
    ax.set_ylabel("In-Degree")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.plot([0, min(cap_out, cap_in)], [0, min(cap_out, cap_in)],
            "k--", linewidth=0.8, alpha=0.4, label="y=x")

fig.suptitle("Out vs In Degree  (dot size = |asymmetry|, gold = extreme)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "asym_scatter.png"), dpi=150)
plt.close()
print("Saved: asym_scatter.png")

# ── 圖 3：箱型圖 |asym| 黑 vs 白 ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot(
    [black["asym"].abs().values, normal["asym"].abs().values],
    patch_artist=True, showfliers=True,
    flierprops=dict(marker="o", markersize=3, alpha=0.3)
)
bp["boxes"][0].set_facecolor("crimson");  bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("seagreen"); bp["boxes"][1].set_alpha(0.7)
for m in bp["medians"]:
    m.set_color("black"); m.set_linewidth(2)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Blacklist (1)", "Normal (0)"], fontsize=11)
ax.set_ylabel("|Asymmetry|")
ax.set_title("|Degree Asymmetry| Boxplot\nBlacklist vs Normal", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "asym_boxplot.png"), dpi=150)
plt.close()
print("Saved: asym_boxplot.png")

# ── 圖 4：stacked bar — asym 分三區間 ────────────────────
def asym_category(s):
    return pd.cut(s, bins=[-1.01, -0.5, 0.5, 1.01],
                  labels=["Receiver\n(in>>out)", "Balanced", "Sender\n(out>>in)"])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, grp, title in zip(axes, [normal, black], ["Normal (0)", "Blacklist (1)"]):
    cats = asym_category(grp["asym"]).value_counts(normalize=True).sort_index()
    colors = ["steelblue", "lightgray", "tomato"]
    bars = ax.bar(cats.index, cats.values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, cats.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Degree Asymmetry Category: Receiver / Balanced / Sender",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "asym_category.png"), dpi=150)
plt.close()
print("Saved: asym_category.png")
