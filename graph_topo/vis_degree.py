"""
只看 train_label 有 label 的用戶（0=正常, 1=黑名單）
分析 In-degree / Out-degree 與黑白名單的關係
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── 路徑 ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "graph_topo", "outputs")
os.makedirs(OUT, exist_ok=True)

# ── 載入 ──────────────────────────────────────────────────
crypto = pd.read_csv(os.path.join(DATA, "crypto_transfer.csv"))
train  = pd.read_csv(os.path.join(DATA, "train_label.csv"))  # user_id, status

# ── 計算 degree ───────────────────────────────────────────
out_deg = crypto.groupby("user_id").size().rename("out_degree")
in_deg  = (crypto[crypto["relation_user_id"].notna()]
           .groupby("relation_user_id").size().rename("in_degree"))

df = (train
      .join(out_deg, on="user_id")
      .join(in_deg,  on="user_id")
      .fillna(0))
df["out_degree"] = df["out_degree"].astype(int)
df["in_degree"]  = df["in_degree"].astype(int)

black  = df[df["status"] == 1]
normal = df[df["status"] == 0]

print(f"正常: {len(normal)}, 黑名單: {len(black)}")
for grp, name in [(black, "黑名單"), (normal, "正常")]:
    print(f"\n{name}:")
    print(grp[["out_degree", "in_degree"]].describe().round(1))

# ── 圖 1：箱型圖（cap 到 90th percentile，去除極端值）────
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, col in zip(axes, ["out_degree", "in_degree"]):
    cap = int(df[col].quantile(0.90))
    bp = ax.boxplot(
        [black[col].clip(upper=cap).values, normal[col].clip(upper=cap).values],
        patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4)
    )
    bp["boxes"][0].set_facecolor("crimson");  bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("seagreen"); bp["boxes"][1].set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black"); median.set_linewidth(2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Blacklist (1)", "Normal (0)"], fontsize=11)
    ax.set_title(f"{col.replace('_',' ').title()}\n(capped at 90th pct = {cap})", fontsize=11)
    ax.set_ylabel("Degree")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.suptitle("Degree Boxplot: Blacklist vs Normal", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "deg_boxplot.png"), dpi=150)
plt.close()
print("Saved: deg_boxplot.png")

# ── 圖 2a：Hexbin 密度圖（正常用戶）────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cap_out = int(df["out_degree"].quantile(0.98))
cap_in  = int(df["in_degree"].quantile(0.98)) + 1

for ax, grp, title, cmap in zip(
    axes,
    [normal, black],
    ["Normal (0)", "Blacklist (1)"],
    ["Greens", "Reds"]
):
    hb = ax.hexbin(
        grp["out_degree"].clip(upper=cap_out),
        grp["in_degree"].clip(upper=cap_in),
        gridsize=30, cmap=cmap, mincnt=1, bins="log"
    )
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("Out-Degree")
    ax.set_ylabel("In-Degree")
    ax.set_title(title, fontsize=12)

fig.suptitle("In-Degree vs Out-Degree (Hexbin density)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "deg_hexbin.png"), dpi=150)
plt.close()
print("Saved: deg_hexbin.png")

# ── 圖 2b：原始散佈圖 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(normal["out_degree"], normal["in_degree"],
           c="seagreen", alpha=0.3, s=10, label="Normal (0)", edgecolors="none")
ax.scatter(black["out_degree"], black["in_degree"],
           c="crimson", alpha=0.6, s=12, label="Blacklist (1)", edgecolors="none")
ax.set_xlabel("Out-Degree")
ax.set_ylabel("In-Degree")
ax.set_title("In-Degree vs Out-Degree", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "deg_scatter.png"), dpi=150)
plt.close()
print("Saved: deg_scatter.png")

# ── 圖 3：密度直方圖 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["out_degree", "in_degree"]):
    cap = df[col].quantile(0.99)
    bins = np.linspace(0, cap, 40)
    ax.hist(normal[col].clip(upper=cap), bins=bins,
            color="seagreen", alpha=0.6, label="Normal (0)", density=True)
    ax.hist(black[col].clip(upper=cap),  bins=bins,
            color="crimson",  alpha=0.6, label="Blacklist (1)", density=True)
    ax.set_title(col.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("Degree (capped at 99th pct)")
    ax.set_ylabel("Density")
    ax.legend()

fig.suptitle("Degree Distribution: Blacklist vs Normal", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "deg_hist.png"), dpi=150)
plt.close()
print("Saved: deg_hist.png")
