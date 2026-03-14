"""
NMF + LightGBM 分類器
使用 NMF 對 User × IP 矩陣進行分解，提取 IP 使用模式特徵
再結合其他特徵用 LightGBM 進行分類
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Microsoft JhengHei"

SCALE = 1e-8
base = "data"

print("=" * 60)
print("NMF + LightGBM 詐騙偵測分類器")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 載入資料
# ─────────────────────────────────────────────
print("\n[1/6] 載入資料...")
twd_transfer = pd.read_csv(f"{base}/twd_transfer.csv")
crypto       = pd.read_csv(f"{base}/crypto_transfer.csv")
trading      = pd.read_csv(f"{base}/usdt_twd_trading.csv")
swap         = pd.read_csv(f"{base}/usdt_swap.csv")
train_label  = pd.read_csv(f"{base}/train_label.csv").rename(columns={"status": "label"})

print(f"  標籤資料：{len(train_label)} 筆")
print(f"  黑名單：{train_label['label'].sum()} 人")
print(f"  正常：{(train_label['label'] == 0).sum()} 人")

# ─────────────────────────────────────────────
# 2. 建立 User × IP 矩陣
# ─────────────────────────────────────────────
print("\n[2/6] 建立 User × IP 矩陣...")

ip_records = pd.concat([
    twd_transfer[["user_id", "source_ip_hash"]],
    crypto[["user_id", "source_ip_hash"]],
    trading[["user_id", "source_ip_hash"]],
], ignore_index=True).dropna(subset=["source_ip_hash"])

# 只保留有 label 的用戶
ip_records = ip_records[ip_records["user_id"].isin(train_label["user_id"])]
ip_records = ip_records.drop_duplicates(subset=["user_id", "source_ip_hash"])

# 篩選共用人數 >= 3 的 IP（降低雜訊）
ip_user_cnt = ip_records.groupby("source_ip_hash")["user_id"].nunique()
shared_ips  = ip_user_cnt[ip_user_cnt >= 3].index
ip_filtered = ip_records[ip_records["source_ip_hash"].isin(shared_ips)]

# 建立 User × IP 矩陣
user_ip_matrix = ip_filtered.pivot_table(
    index="user_id",
    columns="source_ip_hash",
    aggfunc="size",
    fill_value=0
)
user_ip_matrix = (user_ip_matrix > 0).astype(int)

print(f"  矩陣大小：{user_ip_matrix.shape[0]} 用戶 × {user_ip_matrix.shape[1]} IP")
print(f"  稀疏度：{(user_ip_matrix == 0).sum().sum() / user_ip_matrix.size:.2%}")

# ─────────────────────────────────────────────
# 3. NMF 矩陣分解
# ─────────────────────────────────────────────
print("\n[3/6] 執行 NMF 矩陣分解...")

N_COMPONENTS = 20  # 分解成 20 個潛在群體特徵
nmf = NMF(n_components=N_COMPONENTS, init='nndsvd', random_state=42, max_iter=500)
user_features_nmf = nmf.fit_transform(user_ip_matrix)

print(f"  分解維度：{N_COMPONENTS}")
print(f"  重建誤差：{nmf.reconstruction_err_:.2f}")

# 將 NMF 特徵轉成 DataFrame
nmf_df = pd.DataFrame(
    user_features_nmf,
    index=user_ip_matrix.index,
    columns=[f"nmf_{i}" for i in range(N_COMPONENTS)]
)

# ─────────────────────────────────────────────
# 4. 提取其他特徵
# ─────────────────────────────────────────────
print("\n[4/6] 提取其他特徵...")

twd_transfer["amount"] = twd_transfer["ori_samount"] * SCALE
crypto["amount"]       = crypto["ori_samount"] * SCALE
trading["amount"]      = trading["trade_samount"] * SCALE
swap["twd_amount"]     = swap["twd_samount"] * SCALE

# 法幣入金
twd_in = twd_transfer[twd_transfer["kind"] == 0].groupby("user_id").agg({
    "amount": ["sum", "mean", "count"]
}).reset_index()
twd_in.columns = ["user_id", "twd_in_sum", "twd_in_mean", "twd_in_count"]

# 加密幣出金
crypto_out = crypto[crypto["kind"] == 1].groupby("user_id").agg({
    "amount": ["sum", "mean", "count"]
}).reset_index()
crypto_out.columns = ["user_id", "crypto_out_sum", "crypto_out_mean", "crypto_out_count"]

# 交易量
trade_stats = trading.groupby("user_id").agg({
    "amount": ["sum", "mean", "count"]
}).reset_index()
trade_stats.columns = ["user_id", "trade_sum", "trade_mean", "trade_count"]

# IP 共用特徵
ip_share = ip_filtered.groupby("user_id")["source_ip_hash"].agg(["nunique", "count"]).reset_index()
ip_share.columns = ["user_id", "unique_ip_count", "total_ip_records"]

# 合併所有特徵
features = train_label[["user_id", "label"]].copy()
for df in [twd_in, crypto_out, trade_stats, ip_share]:
    features = features.merge(df, on="user_id", how="left")

features = features.fillna(0)

# 加入 NMF 特徵
features = features.merge(nmf_df, left_on="user_id", right_index=True, how="inner")

# 轉換率特徵
features["conversion_rate"] = features["crypto_out_sum"] / (features["twd_in_sum"] + 1e-9)
features["conversion_rate"] = features["conversion_rate"].clip(0, 2)

print(f"  最終特徵數：{features.shape[1] - 2} 個（含 {N_COMPONENTS} 個 NMF 特徵）")
print(f"  樣本數：{len(features)}")

# ─────────────────────────────────────────────
# 5. 訓練 LightGBM
# ─────────────────────────────────────────────
print("\n[5/6] 訓練 LightGBM 分類器...")

X = features.drop(columns=["user_id", "label"])
y = features["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum()  # 處理不平衡
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
)

# ─────────────────────────────────────────────
# 6. 評估結果
# ─────────────────────────────────────────────
print("\n[6/6] 評估結果...")

y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=["正常", "黑名單"]))

print(f"\nROC-AUC：{roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n混淆矩陣：")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
print("\nTop 15 重要特徵：")
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)

print(importance.head(15).to_string(index=False))

# 視覺化 Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
top_features = importance.head(20)
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("Importance (Gain)")
ax.set_title("Top 20 特徵重要性（NMF + LightGBM）")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("nmf_feature_importance.png", dpi=150, bbox_inches="tight")
print("\n特徵重要性圖已儲存：nmf_feature_importance.png")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
