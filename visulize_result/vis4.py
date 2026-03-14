"""
vis4.py - 二分圖共現矩陣（Bipartite Co-occurrence）
根據 process3.md：偵測女巫攻擊（Sybil Attack）
輸出至 visulize_result/outputs4/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALE = 1e-8
base  = os.path.join(os.path.dirname(__file__), "..", "data")

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已儲存：{name}")

# ─────────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────────
print("載入資料...")
twd_transfer = pd.read_csv(f"{base}/twd_transfer.csv")
crypto       = pd.read_csv(f"{base}/crypto_transfer.csv")
trading      = pd.read_csv(f"{base}/usdt_twd_trading.csv")
train_label  = pd.read_csv(f"{base}/train_label.csv").rename(columns={"status": "label"})

black_ids  = set(train_label[train_label["label"] == 1]["user_id"])
normal_ids = set(train_label[train_label["label"] == 0]["user_id"])

# ─────────────────────────────────────────────
# 合併 IP 紀錄
# ─────────────────────────────────────────────
ip_records = pd.concat([
    twd_transfer[["user_id", "source_ip_hash"]],
    crypto[["user_id", "source_ip_hash"]],
    trading[["user_id", "source_ip_hash"]],
], ignore_index=True).dropna(subset=["source_ip_hash"])

# 只保留有 label 的用戶
ip_records = ip_records[ip_records["user_id"].isin(train_label["user_id"])]
ip_records = ip_records.drop_duplicates(subset=["user_id", "source_ip_hash"])

# ─────────────────────────────────────────────
# 篩選：共用人數 >= 4 的 IP
# ─────────────────────────────────────────────
ip_user_cnt = ip_records.groupby("source_ip_hash")["user_id"].nunique()
shared_ips  = ip_user_cnt[ip_user_cnt >= 4].index
ip_filtered = ip_records[ip_records["source_ip_hash"].isin(shared_ips)]

MAX_IP   = 80
MAX_USER = 120

# ─────────────────────────────────────────────
# 共用 IP 排序（依共用人數）
# ─────────────────────────────────────────────
ip_order_all = (
    ip_filtered.groupby("source_ip_hash")["user_id"].nunique()
    .sort_values(ascending=False).index.tolist()
)[:MAX_IP]

black_ip  = ip_filtered[ip_filtered["user_id"].isin(black_ids)]
normal_ip = ip_filtered[ip_filtered["user_id"].isin(normal_ids)]

# ─────────────────────────────────────────────
# 圖1：黑名單 + 正常用戶合併，雙色區分
# ─────────────────────────────────────────────
print("[1/5] 繪製合併雙色鄰接矩陣（黑名單紅、正常綠）...")

combined = ip_filtered[ip_filtered["source_ip_hash"].isin(ip_order_all)].copy()
combined["label"] = combined["user_id"].apply(
    lambda u: 1 if u in black_ids else (0 if u in normal_ids else -1)
)
combined = combined[combined["label"] >= 0]

# 用戶排序：黑名單優先，再依共用 IP 數
user_order_combined = (
    combined.sort_values("label", ascending=False)
    .groupby("user_id")["source_ip_hash"].nunique()
    .sort_values(ascending=False)
    .index.tolist()
)[:MAX_USER * 2]

sub = combined[
    combined["user_id"].isin(user_order_combined) &
    combined["source_ip_hash"].isin(ip_order_all)
]

# 建立兩層矩陣：黑名單=2，正常=1，空=0
pivot_b = black_ip[black_ip["user_id"].isin(user_order_combined) & black_ip["source_ip_hash"].isin(ip_order_all)]\
    .pivot_table(index="user_id", columns="source_ip_hash", aggfunc="size", fill_value=0)
pivot_n = normal_ip[normal_ip["user_id"].isin(user_order_combined) & normal_ip["source_ip_hash"].isin(ip_order_all)]\
    .pivot_table(index="user_id", columns="source_ip_hash", aggfunc="size", fill_value=0)

pivot_b = (pivot_b > 0).astype(int).reindex(index=user_order_combined, columns=ip_order_all, fill_value=0)
pivot_n = (pivot_n > 0).astype(int).reindex(index=user_order_combined, columns=ip_order_all, fill_value=0)

# 合併：黑名單=2，正常=1
matrix_combined = pivot_b * 2 + pivot_n
matrix_combined = matrix_combined.clip(0, 2)

from matplotlib.colors import ListedColormap
cmap3 = ListedColormap(["#F0F0F0", "#2ECC71", "#E74C3C"])

fig, ax = plt.subplots(figsize=(18, 10))
im = ax.imshow(matrix_combined.values, aspect="auto", cmap=cmap3, vmin=0, vmax=2, interpolation="nearest")
ax.set_title(
    f"二分圖鄰接矩陣（黑名單+正常用戶 × 共用IP）\n"
    f"紅=黑名單用戶  綠=正常用戶  X軸：前{len(ip_order_all)}個共用IP  Y軸：前{len(user_order_combined)}個用戶",
    fontsize=11
)
ax.set_xlabel("IP（依共用人數排序）")
ax.set_ylabel("用戶（黑名單優先排序）")
ax.set_xticks([]); ax.set_yticks([])
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="#E74C3C", label="黑名單用戶"),
    mpatches.Patch(color="#2ECC71", label="正常用戶"),
]
ax.legend(handles=legend_handles, loc="upper right")
save(fig, "01a_bipartite_combined_heatmap.png")

# ─────────────────────────────────────────────
# 圖2：左黑名單 / 右正常 雙子圖對比
# ─────────────────────────────────────────────
print("[2/5] 繪製左右對比子圖...")

black_user_order = (
    black_ip.groupby("user_id")["source_ip_hash"].nunique()
    .sort_values(ascending=False).index.tolist()
)[:MAX_USER]

normal_user_order = (
    normal_ip.groupby("user_id")["source_ip_hash"].nunique()
    .sort_values(ascending=False).index.tolist()
)[:MAX_USER]

def build_matrix(df, user_order, ip_order):
    sub = df[df["user_id"].isin(user_order) & df["source_ip_hash"].isin(ip_order)]
    m = sub.pivot_table(index="user_id", columns="source_ip_hash", aggfunc="size", fill_value=0)
    return (m > 0).astype(int).reindex(index=user_order, columns=ip_order, fill_value=0)

mat_b = build_matrix(black_ip,  black_user_order,  ip_order_all)
mat_n = build_matrix(normal_ip, normal_user_order, ip_order_all)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle("二分圖鄰接矩陣對比（左：黑名單用戶  右：正常用戶）\n方塊狀密集區=詐騙農場群體特徵", fontsize=12)

for ax, mat, title, color in zip(
    axes,
    [mat_b, mat_n],
    [f"黑名單用戶（前{len(black_user_order)}個）", f"正常用戶（前{len(normal_user_order)}個）"],
    ["#E74C3C", "#2ECC71"]
):
    cmap2 = ListedColormap(["#F0F0F0", color])
    ax.imshow(mat.values, aspect="auto", cmap=cmap2, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"IP（前{len(ip_order_all)}個）")
    ax.set_ylabel("用戶")
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
save(fig, "01b_bipartite_side_by_side.png")

# ─────────────────────────────────────────────
# 圖3：只保留與黑名單共用同一 IP 的正常用戶（混雜圖）
# ─────────────────────────────────────────────
print("[3/5] 繪製黑名單混雜正常用戶圖...")

# 找出黑名單有使用的 IP
black_used_ips = set(black_ip["source_ip_hash"])

# 正常用戶中，有使用過黑名單 IP 的
mixed_normal = normal_ip[normal_ip["source_ip_hash"].isin(black_used_ips)]
mixed_normal_users = mixed_normal["user_id"].unique()

# 合併黑名單 + 混雜正常用戶
mixed_df = pd.concat([
    black_ip.assign(label=1),
    mixed_normal.assign(label=0),
], ignore_index=True)

# IP 排序：只取黑名單有用的 IP，依共用人數
mixed_ip_order = (
    mixed_df.groupby("source_ip_hash")["user_id"].nunique()
    .sort_values(ascending=False).index.tolist()
)[:MAX_IP]

# 用戶排序：黑名單優先
mixed_user_order = (
    mixed_df.sort_values("label", ascending=False)
    .drop_duplicates("user_id")
    .set_index("user_id")["label"]
    .index.tolist()
)
# 依共用 IP 數再排
mixed_user_cnt = mixed_df.groupby("user_id")["source_ip_hash"].nunique()
black_users_sorted  = mixed_user_cnt[mixed_user_cnt.index.isin(black_ids)].sort_values(ascending=False).index.tolist()[:MAX_USER]
normal_users_sorted = mixed_user_cnt[mixed_user_cnt.index.isin(normal_ids)].sort_values(ascending=False).index.tolist()[:MAX_USER]
mixed_user_order = black_users_sorted + normal_users_sorted

mat_mixed_b = build_matrix(mixed_df[mixed_df["label"]==1], mixed_user_order, mixed_ip_order)
mat_mixed_n = build_matrix(mixed_df[mixed_df["label"]==0], mixed_user_order, mixed_ip_order)

matrix_mixed = mat_mixed_b * 2 + mat_mixed_n
matrix_mixed = matrix_mixed.clip(0, 2)

# 分隔線位置
sep_y = len(black_users_sorted)

fig, ax = plt.subplots(figsize=(18, 10))
ax.imshow(matrix_mixed.values, aspect="auto", cmap=cmap3, vmin=0, vmax=2, interpolation="nearest")
ax.axhline(sep_y - 0.5, color="yellow", linewidth=2, linestyle="--")
ax.set_title(
    f"混雜圖：黑名單用戶 + 與其共用IP的正常用戶\n"
    f"紅=黑名單（{len(black_users_sorted)}人）  綠=混雜正常用戶（{len(normal_users_sorted)}人）  黃線=分隔\n"
    f"綠色出現在紅色密集欄位 = 正常用戶被捲入詐騙 IP 群",
    fontsize=11
)
ax.set_xlabel(f"IP（黑名單使用過，前{len(mixed_ip_order)}個）")
ax.set_ylabel("用戶（上方黑名單 / 下方混雜正常）")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(handles=legend_handles, loc="upper right")
save(fig, "01c_bipartite_mixed_heatmap.png")

# ─────────────────────────────────────────────
# 圖4（原圖1）：純黑名單鄰接矩陣
# ─────────────────────────────────────────────
print("[4/5] 繪製純黑名單鄰接矩陣...")

ip_order_b = (
    black_ip.groupby("source_ip_hash")["user_id"].nunique()
    .sort_values(ascending=False).index.tolist()
)[:MAX_IP]

mat_b_only = build_matrix(black_ip, black_user_order, ip_order_b)

fig, ax = plt.subplots(figsize=(18, 10))
ax.imshow(mat_b_only.values, aspect="auto",
          cmap=ListedColormap(["#F0F0F0", "#E74C3C"]), vmin=0, vmax=1, interpolation="nearest")
ax.set_title(
    f"二分圖鄰接矩陣熱力圖（黑名單用戶 × 共用 IP）\n"
    f"X軸：IP（共用人數≥4，前{len(ip_order_b)}個）  Y軸：黑名單用戶（前{len(black_user_order)}個）\n"
    f"紅色=有使用該IP  方塊狀密集區=詐騙農場群體（女巫攻擊特徵）",
    fontsize=11
)
ax.set_xlabel("IP（依共用人數排序）")
ax.set_ylabel("黑名單用戶（依共用IP數排序）")
ax.set_xticks([]); ax.set_yticks([])
save(fig, "01_bipartite_blacklist_heatmap.png")

# ─────────────────────────────────────────────
# 圖5：黑名單 vs 正常用戶 共用 IP 數分布比較
# ─────────────────────────────────────────────
print("[5/5] 繪製共用 IP 數分布比較...")

ip_share_per_user = (
    ip_filtered.groupby("user_id")["source_ip_hash"].nunique()
    .reset_index().rename(columns={"source_ip_hash": "shared_ip_cnt"})
    .merge(train_label, on="user_id", how="inner")
)

fig, ax = plt.subplots(figsize=(9, 5))
bins = range(1, min(int(ip_share_per_user["shared_ip_cnt"].max()) + 2, 30))
ax.hist(
    ip_share_per_user[ip_share_per_user["label"] == 0]["shared_ip_cnt"].clip(1, 30),
    bins=bins, alpha=0.6, label="正常用戶", color="#2ECC71", density=True
)
ax.hist(
    ip_share_per_user[ip_share_per_user["label"] == 1]["shared_ip_cnt"].clip(1, 30),
    bins=bins, alpha=0.7, label="黑名單用戶", color="#E74C3C", density=True
)
ax.set_title("用戶使用共用 IP（≥4人）的數量分布\n黑名單用戶傾向使用更多共用 IP（女巫攻擊特徵）")
ax.set_xlabel("共用 IP 數量"); ax.set_ylabel("密度"); ax.legend()
save(fig, "02_shared_ip_count_dist.png")

print(f"\n全部圖表已儲存至 visulize_result/outputs4/")
