"""
graph_topo/vis_topo.py
簡單視覺化分析：
1. 節點 In/Out-degree 分布（箱型圖 + 直方圖）
2. 黑名單 vs 正常用戶的 degree 散佈圖
3. sub_kind 分布（0=外部, 1=內部）
4. 高度匯集錢包 Top-N 長條圖
5. 力導向關聯圖（Force-Directed Graph）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ── 載入資料 ──────────────────────────────────────────────
crypto = pd.read_csv("data/crypto_transfer.csv")
train_label = pd.read_csv("data/train_label.csv")
features = pd.read_csv("data/features.csv")

predict_label = pd.read_csv("data/predict_label.csv")

label_map = dict(zip(train_label["user_id"], train_label["status"]))
# predict 用戶標為 2（待預測，尚無 label）
for uid in predict_label["user_id"]:
    if uid not in label_map:
        label_map[uid] = 2

# ── 計算每個 user 的 In/Out-degree ────────────────────────
# Out-degree: user 發出的 crypto 轉帳次數
out_deg = crypto.groupby("user_id").size().rename("out_degree")

# In-degree: 透過 relation_user_id 收到的次數
in_deg = crypto[crypto["relation_user_id"].notna()].groupby("relation_user_id").size().rename("in_degree")

deg_df = pd.DataFrame({"out_degree": out_deg, "in_degree": in_deg}).fillna(0).astype(int)
deg_df["label"] = deg_df.index.map(lambda x: label_map.get(x, -1))  # -1=unknown
deg_df["label_str"] = deg_df["label"].map({1: "Blacklist", 0: "Normal", 2: "To Predict", -1: "Unknown"})

print(f"Total nodes: {len(deg_df)}")
print(deg_df["label_str"].value_counts())
print(deg_df[["out_degree", "in_degree"]].describe())

import os
os.makedirs("graph_topo/outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════
# 圖 1：Out-degree & In-degree 直方圖
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, color, title in zip(
    axes,
    ["out_degree", "in_degree"],
    ["steelblue", "tomato"],
    ["Out-Degree Distribution (# transfers sent)", "In-Degree Distribution (# transfers received)"],
):
    vals = deg_df[col]
    ax.hist(vals[vals < vals.quantile(0.99)], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.axvline(vals.median(), color="black", linestyle="--", linewidth=1, label=f"Median={vals.median():.0f}")
    ax.legend()
fig.suptitle("Node Degree Distributions (crypto_transfer)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("graph_topo/outputs/01_degree_histogram.png", dpi=150)
plt.close()
print("Saved: 01_degree_histogram.png")

# ══════════════════════════════════════════════════════════
# 圖 2：箱型圖 — Blacklist vs Normal vs Unknown
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
groups = ["Blacklist", "Normal", "To Predict"]
colors = ["crimson", "seagreen", "gray"]

for ax, col in zip(axes, ["out_degree", "in_degree"]):
    data = [deg_df[deg_df["label_str"] == g][col].values for g in groups]
    bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(groups)
    ax.set_title(f"{col.replace('_', ' ').title()} by Label", fontsize=11)
    ax.set_ylabel("Degree")
fig.suptitle("Degree Boxplot: Blacklist vs Normal vs Unknown", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("graph_topo/outputs/02_degree_boxplot.png", dpi=150)
plt.close()
print("Saved: 02_degree_boxplot.png")

# ══════════════════════════════════════════════════════════
# 圖 3：散佈圖 — In-degree vs Out-degree（顏色=label）
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
color_map = {"Blacklist": "crimson", "Normal": "seagreen", "To Predict": "lightgray", "Unknown": "black"}
for label, grp in deg_df.groupby("label_str"):
    ax.scatter(grp["out_degree"], grp["in_degree"],
               c=color_map[label], label=label, alpha=0.6, s=20, edgecolors="none")
ax.set_xlabel("Out-Degree (transfers sent)")
ax.set_ylabel("In-Degree (transfers received)")
ax.set_title("In-Degree vs Out-Degree Scatter", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("graph_topo/outputs/03_degree_scatter.png", dpi=150)
plt.close()
print("Saved: 03_degree_scatter.png")

# ══════════════════════════════════════════════════════════
# 圖 4：sub_kind 分布 — 外部(0) vs 內部(1)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
sub_kind_counts = crypto["sub_kind"].value_counts().sort_index()
bars = ax.bar(["External (0)", "Internal (1)"], sub_kind_counts.values,
              color=["steelblue", "darkorange"], edgecolor="white", alpha=0.85)
for bar, val in zip(bars, sub_kind_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:,}", ha="center", va="bottom", fontsize=10)
ax.set_title("sub_kind Distribution (0=External, 1=Internal)", fontsize=12, fontweight="bold")
ax.set_ylabel("Transaction Count")
plt.tight_layout()
plt.savefig("graph_topo/outputs/04_subkind_bar.png", dpi=150)
plt.close()
print("Saved: 04_subkind_bar.png")

# ══════════════════════════════════════════════════════════
# 圖 5：Top-20 高度匯集錢包（to_wallet_hash 入度）
# ══════════════════════════════════════════════════════════
top_wallets = crypto["to_wallet_hash"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(top_wallets)), top_wallets.values, color="tomato", alpha=0.8)
ax.set_yticks(range(len(top_wallets)))
ax.set_yticklabels([w[:16] + "..." for w in top_wallets.index], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of Incoming Transfers")
ax.set_title("Top-20 Wallets by Incoming Transfer Count (Funnel Nodes)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("graph_topo/outputs/05_top_wallets.png", dpi=150)
plt.close()
print("Saved: 05_top_wallets.png")

# ══════════════════════════════════════════════════════════
# 圖 6：Force-Directed Graph（取 Top-200 筆交易）
# ══════════════════════════════════════════════════════════
# 取最活躍的錢包節點周邊交易
top_wallet_set = set(top_wallets.index[:10])
sample = crypto[crypto["to_wallet_hash"].isin(top_wallet_set)].head(200)

G = nx.DiGraph()
for _, row in sample.iterrows():
    src = f"u_{row['user_id']}"
    dst = row["to_wallet_hash"][:12]
    sub = row["sub_kind"]
    G.add_node(src, kind="user", sub_kind=sub)
    G.add_node(dst, kind="wallet")
    G.add_edge(src, dst)

# 節點大小 = degree
node_sizes = []
node_colors = []
for n in G.nodes():
    deg = G.degree(n)
    node_sizes.append(max(30, deg * 40))
    if G.nodes[n].get("kind") == "wallet":
        node_colors.append("tomato")
    elif G.nodes[n].get("sub_kind") == 1:
        node_colors.append("darkorange")
    else:
        node_colors.append("steelblue")

fig, ax = plt.subplots(figsize=(12, 9))
pos = nx.spring_layout(G, seed=42, k=0.5)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=8,
                       edge_color="gray", ax=ax)
# 只標 wallet 節點
wallet_labels = {n: n for n in G.nodes() if G.nodes[n].get("kind") == "wallet"}
nx.draw_networkx_labels(G, pos, labels=wallet_labels, font_size=7, ax=ax)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="tomato", label="Wallet (funnel node)"),
    Patch(facecolor="darkorange", label="User - Internal (sub_kind=1)"),
    Patch(facecolor="steelblue", label="User - External (sub_kind=0)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
ax.set_title("Force-Directed Graph: Top-10 Funnel Wallets & Connected Users", fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("graph_topo/outputs/06_force_directed_graph.png", dpi=150)
plt.close()
print("Saved: 06_force_directed_graph.png")

print("\nAll charts saved to graph_topo/outputs/")
