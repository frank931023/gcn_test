"""
vis2.py - 資料前處理、清洗與視覺化
根據 process.md 的分析方向，快速判斷各特徵對黑名單識別的幫助
輸出圖表至 visulize_result/outputs2/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import warnings
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# 支援中文字型（Windows）
matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALE = 1e-8  # 金額還原倍率

# ─────────────────────────────────────────────
# 1. 載入資料
# ─────────────────────────────────────────────
print("載入資料...")
base = os.path.join(os.path.dirname(__file__), "..", "data")

user_info   = pd.read_csv(f"{base}/user_info.csv")
twd_transfer = pd.read_csv(f"{base}/twd_transfer.csv")
crypto      = pd.read_csv(f"{base}/crypto_transfer.csv")
trading     = pd.read_csv(f"{base}/usdt_twd_trading.csv")
swap        = pd.read_csv(f"{base}/usdt_swap.csv")
train_label = pd.read_csv(f"{base}/train_label.csv")

# ─────────────────────────────────────────────
# 2. 資料前處理 & 清洗
# ─────────────────────────────────────────────
print("前處理中...")

# 2-1 金額還原 (×1e-8)
twd_transfer["amount"]  = twd_transfer["ori_samount"] * SCALE
crypto["amount"]        = crypto["ori_samount"] * SCALE
trading["trade_amount"] = trading["trade_samount"] * SCALE
trading["twd_rate"]     = trading["twd_srate"] * SCALE
swap["twd_amount"]      = swap["twd_samount"] * SCALE
swap["currency_amount"] = swap["currency_samount"] * SCALE

# 2-2 時間欄位統一轉為 datetime（UTC）
def to_dt(df, col):
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

for df, col in [
    (user_info,    "confirmed_at"),
    (user_info,    "level1_finished_at"),
    (user_info,    "level2_finished_at"),
    (twd_transfer, "created_at"),
    (crypto,       "created_at"),
    (trading,      "updated_at"),
    (swap,         "created_at"),
]:
    to_dt(df, col)

# 2-3 合併標籤（黑名單 status=1，正常 status=0）
label = train_label.rename(columns={"status": "label"})

def merge_label(df):
    return df.merge(label, on="user_id", how="inner")

twd_l     = merge_label(twd_transfer)
crypto_l  = merge_label(crypto)
trading_l = merge_label(trading)
swap_l    = merge_label(swap)

# ─────────────────────────────────────────────
# 3. 特徵工程（User-level）
# ─────────────────────────────────────────────
print("特徵工程中...")

user = label.copy()

# A. KYC 速度（level1 → level2 天數）
ui = user_info.copy()
ui["kyc_days"] = (
    ui["level2_finished_at"] - ui["level1_finished_at"]
).dt.total_seconds() / 86400
user = user.merge(
    ui[["user_id", "kyc_days", "career", "income_source", "user_source"]],
    on="user_id", how="left"
)

# B. 法幣出入金不平衡度
twd_in  = twd_l[twd_l["kind"] == 0].groupby("user_id")["amount"].sum().rename("twd_in")
twd_out = twd_l[twd_l["kind"] == 1].groupby("user_id")["amount"].sum().rename("twd_out")
user = user.merge(twd_in, on="user_id", how="left").merge(twd_out, on="user_id", how="left")
user[["twd_in", "twd_out"]] = user[["twd_in", "twd_out"]].fillna(0)
user["twd_imbalance"] = (user["twd_out"] - user["twd_in"]) / (user["twd_in"] + 1e-9)

# C. 加密貨幣出入金不平衡度
c_in  = crypto_l[crypto_l["kind"] == 0].groupby("user_id")["amount"].sum().rename("crypto_in")
c_out = crypto_l[crypto_l["kind"] == 1].groupby("user_id")["amount"].sum().rename("crypto_out")
user = user.merge(c_in, on="user_id", how="left").merge(c_out, on="user_id", how="left")
user[["crypto_in", "crypto_out"]] = user[["crypto_in", "crypto_out"]].fillna(0)
user["crypto_imbalance"] = (user["crypto_out"] - user["crypto_in"]) / (user["crypto_in"] + 1e-9)

# D. 交易行為特徵
mkt = trading_l.groupby("user_id").agg(
    market_ratio=("is_market", "mean"),
    api_ratio=("source", lambda x: (x == 2).mean()),
    trade_count=("id", "count"),
).reset_index()
user = user.merge(mkt, on="user_id", how="left")

# E. 深夜交易比例（22:00–06:00）
trading_l["hour"] = trading_l["updated_at"].dt.hour
trading_l["is_night"] = trading_l["hour"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)
night = trading_l.groupby("user_id")["is_night"].mean().rename("night_trade_ratio").reset_index()
user = user.merge(night, on="user_id", how="left")

# F. 內部轉帳次數（sub_kind=1）
internal = (
    crypto_l[crypto_l["sub_kind"] == 1]
    .groupby("user_id")["id"].count()
    .rename("internal_transfer_cnt").reset_index()
)
user = user.merge(internal, on="user_id", how="left")
user["internal_transfer_cnt"] = user["internal_transfer_cnt"].fillna(0)

# G. IP 共用度（同一 IP 有多少 user）
ip_user_cnt = twd_transfer.groupby("source_ip_hash")["user_id"].nunique().rename("ip_user_cnt")
twd_ip = twd_transfer[["user_id", "source_ip_hash"]].drop_duplicates()
twd_ip = twd_ip.merge(ip_user_cnt, on="source_ip_hash")
max_ip_share = twd_ip.groupby("user_id")["ip_user_cnt"].max().rename("max_ip_share").reset_index()
user = user.merge(max_ip_share, on="user_id", how="left")
user["max_ip_share"] = user["max_ip_share"].fillna(1)

print(f"特徵表完成，共 {len(user)} 筆，黑名單 {user['label'].sum()} 筆")

# ─────────────────────────────────────────────
# 4. 視覺化
# ─────────────────────────────────────────────
print("繪圖中...")

black  = user[user["label"] == 1]
normal = user[user["label"] == 0]

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已儲存：{name}")

features = {
    "kyc_days":              "KYC 完成天數",
    "twd_imbalance":         "法幣出入金不平衡度",
    "crypto_imbalance":      "加密貨幣出入金不平衡度",
    "market_ratio":          "市價單比例",
    "api_ratio":             "API 下單比例",
    "night_trade_ratio":     "深夜交易比例",
    "internal_transfer_cnt": "內部轉帳次數",
    "max_ip_share":          "IP 最大共用人數",
}

# ── 圖1：各特徵箱型圖（黑 vs 正常）──
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("各特徵分布：黑名單 vs 正常用戶", fontsize=14)
for ax, (col, title) in zip(axes.flatten(), features.items()):
    lo, hi = user[col].quantile(0.01), user[col].quantile(0.99)
    data = [normal[col].dropna().clip(lo, hi), black[col].dropna().clip(lo, hi)]
    ax.boxplot(data, labels=["正常", "黑名單"], patch_artist=True,
               boxprops=dict(facecolor="#AED6F1"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(title, fontsize=10)
plt.tight_layout()
save(fig, "01_feature_boxplot.png")

# ── 圖2：黑名單比例 by 註冊來源 ──
fig, ax = plt.subplots(figsize=(6, 4))
user["user_source_label"] = user["user_source"].map({0: "WEB", 1: "APP"}).fillna("其他")
grp = user.groupby("user_source_label")["label"].mean().reset_index()
bars = ax.bar(grp["user_source_label"], grp["label"] * 100, color=["#5DADE2", "#E74C3C"])
ax.set_title("各註冊來源的黑名單比例")
ax.set_ylabel("黑名單比例 (%)")
for bar, v in zip(bars, grp["label"]):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.2, f"{v*100:.1f}%", ha="center")
save(fig, "02_blacklist_by_source.png")

# ── 圖3：KYC 速度分布 ──
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 365, 50)
ax.hist(normal["kyc_days"].dropna().clip(0, 365), bins=bins, alpha=0.6, label="正常",  color="#2ECC71", density=True)
ax.hist(black["kyc_days"].dropna().clip(0, 365),  bins=bins, alpha=0.7, label="黑名單", color="#E74C3C", density=True)
ax.set_title("KYC 完成天數分布（level1→level2）")
ax.set_xlabel("天數"); ax.set_ylabel("密度"); ax.legend()
save(fig, "03_kyc_days_dist.png")

# ── 圖4：法幣出入金不平衡度分布 ──
fig, ax = plt.subplots(figsize=(8, 4))
clip_val = 5
ax.hist(normal["twd_imbalance"].dropna().clip(-clip_val, clip_val), bins=60, alpha=0.6, label="正常",  color="#2ECC71", density=True)
ax.hist(black["twd_imbalance"].dropna().clip(-clip_val, clip_val),  bins=60, alpha=0.7, label="黑名單", color="#E74C3C", density=True)
ax.set_title("法幣出入金不平衡度分布（快進快出指標）")
ax.set_xlabel("(出金 - 入金) / 入金"); ax.set_ylabel("密度"); ax.legend()
save(fig, "04_twd_imbalance_dist.png")

# ── 圖5：深夜交易比例 vs 市價單比例（散點圖）──
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(normal["night_trade_ratio"].fillna(0), normal["market_ratio"].fillna(0),
           alpha=0.3, s=15, label="正常", color="#2ECC71")
ax.scatter(black["night_trade_ratio"].fillna(0),  black["market_ratio"].fillna(0),
           alpha=0.6, s=30, label="黑名單", color="#E74C3C", marker="x")
ax.set_title("深夜交易比例 vs 市價單比例")
ax.set_xlabel("深夜交易比例"); ax.set_ylabel("市價單比例"); ax.legend()
save(fig, "05_night_vs_market.png")

# ── 圖6：IP 共用人數分布 ──
fig, ax = plt.subplots(figsize=(8, 4))
max_bin = min(int(user["max_ip_share"].max()) + 2, 30)
bins = range(1, max_bin)
ax.hist(normal["max_ip_share"].clip(1, 30), bins=bins, alpha=0.6, label="正常",  color="#2ECC71", density=True)
ax.hist(black["max_ip_share"].clip(1, 30),  bins=bins, alpha=0.7, label="黑名單", color="#E74C3C", density=True)
ax.set_title("同一 IP 最大共用人數分布")
ax.set_xlabel("共用人數"); ax.set_ylabel("密度"); ax.legend()
save(fig, "06_ip_share_dist.png")

# ── 圖7：特徵相關性熱圖 ──
feat_cols = list(features.keys()) + ["label"]
corr = user[feat_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            xticklabels=[features.get(c, c) for c in feat_cols],
            yticklabels=[features.get(c, c) for c in feat_cols], ax=ax)
ax.set_title("特徵相關性熱圖（含 label）")
plt.tight_layout()
save(fig, "07_feature_correlation.png")

# ── 圖8：各特徵識別力（Mann-Whitney AUC）──
results = []
for col in features:
    a = black[col].dropna()
    b = normal[col].dropna()
    if len(a) > 0 and len(b) > 0:
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        auc = stat / (len(a) * len(b))
        auc = max(auc, 1 - auc)
        results.append({"feature": features[col], "AUC": auc, "p_value": p})

res_df = pd.DataFrame(results).sort_values("AUC", ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#E74C3C" if v >= 0.6 else "#F39C12" if v >= 0.55 else "#95A5A6" for v in res_df["AUC"]]
ax.barh(res_df["feature"], res_df["AUC"], color=colors)
ax.axvline(0.5, color="gray",   linestyle="--", label="隨機基準 (0.5)")
ax.axvline(0.6, color="orange", linestyle="--", alpha=0.7, label="中等識別力 (0.6)")
ax.set_title("各特徵對黑名單的識別力（Mann-Whitney AUC）")
ax.set_xlabel("AUC"); ax.legend()
plt.tight_layout()
save(fig, "08_feature_auc.png")

# ── 圖9：交易時段熱圖（黑 vs 正常）──
trading_l["weekday"] = trading_l["updated_at"].dt.dayofweek
pivot_black  = trading_l[trading_l["label"] == 1].groupby(["weekday", "hour"])["id"].count().unstack(fill_value=0)
pivot_normal = trading_l[trading_l["label"] == 0].groupby(["weekday", "hour"])["id"].count().unstack(fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for ax, pivot, title in zip(axes, [pivot_normal, pivot_black], ["正常用戶", "黑名單用戶"]):
    pivot = pivot.reindex(range(7), fill_value=0)
    pivot.index = days
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label": "交易次數"})
    ax.set_title(f"交易時段熱圖 - {title}")
    ax.set_xlabel("小時"); ax.set_ylabel("星期")
plt.tight_layout()
save(fig, "09_trade_hour_heatmap.png")

print("\n全部圖表已儲存至 visulize_result/outputs2/")
print(res_df[["feature", "AUC", "p_value"]].sort_values("AUC", ascending=False).to_string(index=False))
