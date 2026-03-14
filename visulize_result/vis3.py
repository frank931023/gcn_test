"""
vis3.py - 進階視覺化分析
根據 process2.md 的四個方向：
  1. 資金匯集網路圖 (Hub-and-Spoke Graph)
  2. 資金流轉桑基圖 (Sankey Diagram)
  3. IP 共現散佈圖 (IP Co-occurrence Scatter)
  4. 24小時時序行為熱力圖 (Temporal Heatmap)
輸出至 visulize_result/outputs3/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import os
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALE = 1e-8
base  = os.path.join(os.path.dirname(__file__), "..", "data")

# ─────────────────────────────────────────────
# 載入 & 前處理
# ─────────────────────────────────────────────
print("載入資料...")
user_info    = pd.read_csv(f"{base}/user_info.csv")
twd_transfer = pd.read_csv(f"{base}/twd_transfer.csv")
crypto       = pd.read_csv(f"{base}/crypto_transfer.csv")
trading      = pd.read_csv(f"{base}/usdt_twd_trading.csv")
swap         = pd.read_csv(f"{base}/usdt_swap.csv")
train_label  = pd.read_csv(f"{base}/train_label.csv").rename(columns={"status": "label"})

def to_dt(df, col):
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

for df, col in [
    (twd_transfer, "created_at"),
    (crypto,       "created_at"),
    (trading,      "updated_at"),
]:
    to_dt(df, col)

twd_transfer["amount"] = twd_transfer["ori_samount"] * SCALE
crypto["amount"]       = crypto["ori_samount"] * SCALE
trading["amount"]      = trading["trade_samount"] * SCALE
swap["twd_amount"]     = swap["twd_samount"] * SCALE

def merge_label(df):
    return df.merge(train_label, on="user_id", how="inner")

twd_l     = merge_label(twd_transfer)
crypto_l  = merge_label(crypto)
trading_l = merge_label(trading)

black_ids  = set(train_label[train_label["label"] == 1]["user_id"])
normal_ids = set(train_label[train_label["label"] == 0]["user_id"])

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已儲存：{name}")

# ═══════════════════════════════════════════════════════
# 圖1：資金匯集網路圖 (Hub-and-Spoke)
# ═══════════════════════════════════════════════════════
print("\n[1/4] 繪製資金匯集網路圖...")

# 取內部轉帳（sub_kind=1）且有 relation_user_id 的紀錄
internal = crypto_l[
    (crypto_l["sub_kind"] == 1) & (crypto_l["relation_user_id"].notna())
].copy()
internal["relation_user_id"] = internal["relation_user_id"].astype(int)

# 找出最多人匯入的 hub（前 10 個 relation_user_id）
top_hubs = (
    internal.groupby("relation_user_id")["user_id"]
    .nunique()
    .nlargest(10)
    .index.tolist()
)

# 取前 3 個 hub 畫圖（避免太擁擠）
G = nx.DiGraph()
hub_edges = internal[internal["relation_user_id"].isin(top_hubs[:10])]

for _, row in hub_edges.iterrows():
    src = f"U{row['user_id']}"
    dst = f"U{row['relation_user_id']}"
    G.add_node(src, is_black=(row["user_id"] in black_ids))
    G.add_node(dst, is_black=(row["relation_user_id"] in black_ids))
    G.add_edge(src, dst, weight=row["amount"])

fig, ax = plt.subplots(figsize=(14, 10))
if len(G.nodes) > 0:
    pos = nx.spring_layout(G, seed=42, k=1.5)
    black_nodes  = [n for n, d in G.nodes(data=True) if d.get("is_black")]
    normal_nodes = [n for n, d in G.nodes(data=True) if not d.get("is_black")]

    nx.draw_networkx_nodes(G, pos, nodelist=black_nodes,  node_color="#E74C3C", node_size=200, ax=ax, label="黑名單用戶")
    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color="#AED6F1", node_size=100, ax=ax, label="正常用戶")
    nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, arrowsize=10,
                           edge_color="#888888", ax=ax)
    ax.legend(loc="upper left", fontsize=9)
else:
    ax.text(0.5, 0.5, "無內部轉帳資料", ha="center", va="center", transform=ax.transAxes)

ax.set_title("資金匯集網路圖（內部轉帳 Hub-and-Spoke）\n紅色=黑名單用戶，藍色=正常用戶", fontsize=12)
ax.axis("off")
save(fig, "01_hub_spoke_network.png")

# ═══════════════════════════════════════════════════════
# 圖2：資金流轉桑基圖（用 alluvial 近似）
# ═══════════════════════════════════════════════════════
print("[2/4] 繪製資金流轉圖...")

# 計算每個用戶三個階段的總金額
twd_in_amt   = twd_l[twd_l["kind"] == 0].groupby(["user_id", "label"])["amount"].sum().reset_index().rename(columns={"amount": "twd_in"})
trade_amt    = trading_l.groupby(["user_id", "label"])["amount"].sum().reset_index().rename(columns={"amount": "trade"})
crypto_out_amt = crypto_l[crypto_l["kind"] == 1].groupby(["user_id", "label"])["amount"].sum().reset_index().rename(columns={"amount": "crypto_out"})

flow = twd_in_amt.merge(trade_amt[["user_id", "trade"]], on="user_id", how="outer")
flow = flow.merge(crypto_out_amt[["user_id", "crypto_out"]], on="user_id", how="outer")
flow = flow.fillna(0)

# 計算轉換率（出金 / 入金）
flow["conversion"] = flow["crypto_out"] / (flow["twd_in"] + 1e-9)
flow["conversion"] = flow["conversion"].clip(0, 2)

# 用箱型圖 + 散點圖近似桑基圖的概念
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("資金流轉三階段金額分布（黑名單 vs 正常）", fontsize=13)

stages = [("twd_in", "法幣入金 (TWD)"), ("trade", "USDT 交易量"), ("crypto_out", "加密幣出金")]
for ax, (col, title) in zip(axes, stages):
    data_n = flow[flow["label"] == 0][col].clip(0, flow[col].quantile(0.95))
    data_b = flow[flow["label"] == 1][col].clip(0, flow[col].quantile(0.95))
    ax.boxplot([data_n, data_b], labels=["正常", "黑名單"], patch_artist=True,
               boxprops=dict(facecolor="#AED6F1"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("金額 (TWD)")

plt.tight_layout()
save(fig, "02a_fund_flow_stages.png")

# 轉換率分布
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 2, 60)
ax.hist(flow[flow["label"] == 0]["conversion"], bins=bins, alpha=0.6, label="正常",  color="#2ECC71", density=True)
ax.hist(flow[flow["label"] == 1]["conversion"], bins=bins, alpha=0.7, label="黑名單", color="#E74C3C", density=True)
ax.axvline(1.0, color="black", linestyle="--", label="100% 轉換率")
ax.set_title("資金轉換率分布（加密幣出金 / 法幣入金）\n黑名單用戶傾向 100% 轉出")
ax.set_xlabel("轉換率"); ax.set_ylabel("密度"); ax.legend()
save(fig, "02b_fund_conversion_rate.png")

# ═══════════════════════════════════════════════════════
# 圖3：IP 共現散佈圖
# ═══════════════════════════════════════════════════════
print("[3/4] 繪製 IP 共現散佈圖...")

# 合併三個來源的 IP 紀錄
ip_records = pd.concat([
    twd_l[["user_id", "source_ip_hash", "label"]].assign(src="法幣轉帳"),
    crypto_l[["user_id", "source_ip_hash", "label"]].assign(src="加密轉帳"),
    trading_l[["user_id", "source_ip_hash", "label"]].assign(src="交易"),
], ignore_index=True).dropna(subset=["source_ip_hash"])

# 取共用人數 >= 4 人的所有 IP
ip_user_cnt = ip_records.groupby("source_ip_hash")["user_id"].nunique()
top_ips = ip_user_cnt[ip_user_cnt >= 4].sort_values(ascending=False).index.tolist()

plot_data = ip_records[ip_records["source_ip_hash"].isin(top_ips)].drop_duplicates(
    subset=["user_id", "source_ip_hash"]
)

# IP 依共用人數排序編號，user_id 直接用數值
ip_idx   = {ip: i for i, ip in enumerate(top_ips)}
plot_data = plot_data.copy()
plot_data["ip_x"]   = plot_data["source_ip_hash"].map(ip_idx)
plot_data["user_y"] = plot_data["user_id"]

fig, ax = plt.subplots(figsize=(16, 9))
colors = plot_data["label"].map({0: "#2ECC71", 1: "#E74C3C"})
ax.scatter(plot_data["ip_x"], plot_data["user_y"], c=colors, s=10, alpha=0.6)
ax.set_title(f"IP 共現散佈圖（共用人數 ≥ 5 的所有 IP，共 {len(top_ips)} 個）\n垂直密集線 = 多人共用同一 IP（詐騙農場特徵）", fontsize=11)
ax.set_xlabel(f"IP 索引（依共用人數排序，共 {len(top_ips)} 個 IP）")
ax.set_ylabel("用戶 ID")
legend_handles = [
    mpatches.Patch(color="#2ECC71", label="正常用戶"),
    mpatches.Patch(color="#E74C3C", label="黑名單用戶"),
]
ax.legend(handles=legend_handles)
save(fig, "03_ip_cooccurrence_scatter.png")

# IP 共用人數分布（黑 vs 正常）
ip_share = ip_records.merge(ip_user_cnt.rename("ip_user_cnt"), on="source_ip_hash")
max_share = ip_share.groupby(["user_id", "label"])["ip_user_cnt"].max().reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
bins = range(1, min(int(max_share["ip_user_cnt"].max()) + 2, 40))
ax.hist(max_share[max_share["label"] == 0]["ip_user_cnt"].clip(1, 40),
        bins=bins, alpha=0.6, label="正常",  color="#2ECC71", density=True)
ax.hist(max_share[max_share["label"] == 1]["ip_user_cnt"].clip(1, 40),
        bins=bins, alpha=0.7, label="黑名單", color="#E74C3C", density=True)
ax.set_title("用戶最大 IP 共用人數分布")
ax.set_xlabel("共用人數"); ax.set_ylabel("密度"); ax.legend()
save(fig, "03b_ip_share_dist.png")

# ═══════════════════════════════════════════════════════
# 圖4：24小時時序行為熱力圖
# ═══════════════════════════════════════════════════════
print("[4/4] 繪製時序行為熱力圖...")

trading_l["hour"]    = trading_l["updated_at"].dt.hour
trading_l["weekday"] = trading_l["updated_at"].dt.dayofweek
crypto_l["hour"]     = crypto_l["created_at"].dt.hour
crypto_l["weekday"]  = crypto_l["created_at"].dt.dayofweek

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def make_heatmap(df, label_val, title, ax):
    sub = df[df["label"] == label_val]
    pivot = sub.groupby(["weekday", "hour"])["id"].count().unstack(fill_value=0)
    pivot = pivot.reindex(range(7), fill_value=0)
    pivot.index = days
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    # 正規化為每天比例（看相對分布）
    pivot_norm = pivot.div(pivot.sum(axis=1) + 1e-9, axis=0)
    sns.heatmap(pivot_norm, ax=ax, cmap="YlOrRd", cbar_kws={"label": "比例"},
                linewidths=0.3, linecolor="white")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("小時 (0-23)"); ax.set_ylabel("星期")

# 交易熱力圖
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("USDT 交易時段熱力圖（每日相對比例）\n機器人特徵：24小時均勻分布或棋盤狀", fontsize=12)
make_heatmap(trading_l, 0, "正常用戶", axes[0])
make_heatmap(trading_l, 1, "黑名單用戶", axes[1])
plt.tight_layout()
save(fig, "04a_trading_heatmap.png")

# 加密轉帳熱力圖
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("加密貨幣轉帳時段熱力圖（每日相對比例）", fontsize=12)
make_heatmap(crypto_l, 0, "正常用戶", axes[0])
make_heatmap(crypto_l, 1, "黑名單用戶", axes[1])
plt.tight_layout()
save(fig, "04b_crypto_heatmap.png")

# API 下單（source=2）vs 手動下單 時段比較
trading_l["is_api"] = (trading_l["source"] == 2).astype(int)
api_trades = trading_l[trading_l["is_api"] == 1]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("API 自動下單時段熱力圖（source=2）\n棋盤狀 = 定時自動化腳本特徵", fontsize=12)
for ax, label_val, title in zip(axes, [0, 1], ["正常用戶 API 下單", "黑名單用戶 API 下單"]):
    sub = api_trades[api_trades["label"] == label_val]
    if len(sub) > 0:
        pivot = sub.groupby(["weekday", "hour"])["id"].count().unstack(fill_value=0)
        pivot = pivot.reindex(range(7), fill_value=0)
        pivot.index = days
        pivot = pivot.reindex(columns=range(24), fill_value=0)
        pivot_norm = pivot.div(pivot.sum(axis=1) + 1e-9, axis=0)
        sns.heatmap(pivot_norm, ax=ax, cmap="Blues", cbar_kws={"label": "比例"},
                    linewidths=0.3, linecolor="white")
    else:
        ax.text(0.5, 0.5, "無資料", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("小時 (0-23)"); ax.set_ylabel("星期")
plt.tight_layout()
save(fig, "04c_api_trading_heatmap.png")

print(f"\n全部圖表已儲存至 visulize_result/outputs3/")
