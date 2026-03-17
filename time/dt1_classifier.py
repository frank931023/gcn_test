"""
ΔT1 分類器實驗
用 KYC Level2 完成 → 首次入金 的時間差（天）做黑白名單分類
方法：
  1. 多個固定 threshold（規則式）
  2. Logistic Regression（單特徵）
  3. 最佳 threshold 搜尋（Youden's J / F1 / Precision-Recall）
  4. 輸出 MD 報告 + 圖表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

OUT_DIR = "time/outputs_dt1_clf"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 載入資料 ─────────────────────────────────────────────────────
print("Loading data...")
user_info       = pd.read_csv("data/user_info.csv",       parse_dates=["level2_finished_at"])
twd_transfer    = pd.read_csv("data/twd_transfer.csv",    parse_dates=["created_at"])
labels          = pd.read_csv("data/train_label.csv")

user_info = user_info.merge(labels, on="user_id", how="left")
user_info["status"] = user_info["status"].fillna(0).astype(int)

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
df = df[df["dt1_days"] >= 0].copy()

X = df["dt1_days"].values.reshape(-1, 1)
y = df["status"].values

print(f"Total samples: {len(df):,}  Blacklist: {y.sum():,} ({y.mean()*100:.2f}%)")

# ════════════════════════════════════════════════════════════════
# 1. 固定 Threshold 規則式分類
# ════════════════════════════════════════════════════════════════
thresholds = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
thresh_results = []

for t in thresholds:
    pred = (df["dt1_days"] <= t).astype(int).values
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec  = precision_score(y, pred, zero_division=0)
    rec   = recall_score(y, pred, zero_division=0)
    f1    = f1_score(y, pred, zero_division=0)
    spec  = tn / (tn + fp) if (tn+fp) > 0 else 0
    acc   = (tp + tn) / len(y)
    thresh_results.append({
        "threshold_days": t,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": prec, "Recall": rec, "F1": f1,
        "Specificity": spec, "Accuracy": acc,
        "Flagged_pct": pred.mean() * 100
    })

thresh_df = pd.DataFrame(thresh_results)

# ════════════════════════════════════════════════════════════════
# 2. Logistic Regression（5-fold CV）
# ════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob_cv = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

auc_roc = roc_auc_score(y, y_prob_cv)
ap      = average_precision_score(y, y_prob_cv)
print(f"LR CV AUC-ROC: {auc_roc:.4f}  AP: {ap:.4f}")

# 最佳 threshold（Youden's J）
fpr, tpr, roc_thresholds = roc_curve(y, y_prob_cv)
youden_j = tpr - fpr
best_idx_youden = np.argmax(youden_j)
best_thresh_youden = roc_thresholds[best_idx_youden]

# 最佳 threshold（F1）
prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y, y_prob_cv)
f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
best_idx_f1 = np.argmax(f1_arr)
best_thresh_f1 = pr_thresholds[best_idx_f1]

# LR 對應的 dt1 threshold（反推）
lr.fit(X_scaled, y)
# 找 prob=0.5 對應的 dt1 值
dt1_range = np.linspace(0, 1500, 10000).reshape(-1, 1)
probs = lr.predict_proba(scaler.transform(dt1_range))[:, 1]
# Youden threshold 對應的 dt1
dt1_youden = dt1_range[np.argmin(np.abs(probs - best_thresh_youden))][0]
dt1_f1     = dt1_range[np.argmin(np.abs(probs - best_thresh_f1))][0]
dt1_50     = dt1_range[np.argmin(np.abs(probs - 0.5))][0]

print(f"LR best threshold (Youden): prob={best_thresh_youden:.3f} -> dt1={dt1_youden:.1f}d")
print(f"LR best threshold (F1):     prob={best_thresh_f1:.3f} -> dt1={dt1_f1:.1f}d")
print(f"LR threshold (prob=0.5):    dt1={dt1_50:.1f}d")

# ════════════════════════════════════════════════════════════════
# 3. 圖表輸出
# ════════════════════════════════════════════════════════════════
COLOR_N = "#4C9BE8"; COLOR_B = "#E84C4C"

def save(fig, name):
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")

# 圖A：Threshold vs Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#F8F9FA")
for ax in axes: ax.set_facecolor("#FAFAFA")

ax = axes[0]
ax.plot(thresh_df["threshold_days"], thresh_df["Precision"],  "o-", color="#E84C4C", label="Precision")
ax.plot(thresh_df["threshold_days"], thresh_df["Recall"],     "s-", color="#4C9BE8", label="Recall")
ax.plot(thresh_df["threshold_days"], thresh_df["F1"],         "^-", color="#2CA02C", label="F1")
ax.plot(thresh_df["threshold_days"], thresh_df["Specificity"],"D-", color="#FF7F0E", label="Specificity")
ax.set_xlabel("Threshold (days)"); ax.set_ylabel("Score")
ax.set_title("Rule-Based Threshold: Precision / Recall / F1 / Specificity")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax2 = ax.twinx()
ax.bar(thresh_df["threshold_days"], thresh_df["Flagged_pct"],
       width=8, color="#9467BD", alpha=0.5, label="Flagged %")
ax2.plot(thresh_df["threshold_days"], thresh_df["Accuracy"], "o-", color="#8C564B", linewidth=2, label="Accuracy")
ax.set_xlabel("Threshold (days)"); ax.set_ylabel("Flagged Users (%)", color="#9467BD")
ax2.set_ylabel("Accuracy", color="#8C564B")
ax.set_title("Rule-Based Threshold: Flagged Rate & Accuracy")
ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)
save(fig, "A_threshold_metrics")

# 圖B：ROC Curve
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
ax.plot(fpr, tpr, color=COLOR_B, linewidth=2.5, label=f"LR (AUC={auc_roc:.3f})")
ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5)
ax.scatter(fpr[best_idx_youden], tpr[best_idx_youden],
           color="orange", s=120, zorder=5, label=f"Best Youden (dt1<={dt1_youden:.0f}d)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Logistic Regression (5-fold CV)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "B_roc_curve")

# 圖C：Precision-Recall Curve
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
ax.plot(rec_arr, prec_arr, color=COLOR_B, linewidth=2.5, label=f"LR (AP={ap:.3f})")
ax.axhline(y.mean(), color="gray", linestyle="--", linewidth=1, label=f"Baseline ({y.mean()*100:.1f}%)")
ax.scatter(rec_arr[best_idx_f1], prec_arr[best_idx_f1],
           color="orange", s=120, zorder=5, label=f"Best F1 (dt1<={dt1_f1:.0f}d)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - Logistic Regression (5-fold CV)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "C_pr_curve")

# 圖D：LR Probability vs dt1
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
ax.plot(dt1_range.flatten(), probs, color=COLOR_B, linewidth=2.5, label="P(blacklist | dt1)")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="prob=0.5")
ax.axvline(dt1_50,     color="orange", linestyle="--", linewidth=1.5, label=f"prob=0.5 → {dt1_50:.0f}d")
ax.axvline(dt1_youden, color="green",  linestyle=":",  linewidth=1.5, label=f"Youden → {dt1_youden:.0f}d")
ax.axvline(dt1_f1,     color="purple", linestyle="-.", linewidth=1.5, label=f"Best F1 → {dt1_f1:.0f}d")
ax.set_xlabel("ΔT1: KYC Level2 -> First Deposit (days)")
ax.set_ylabel("P(Blacklist)")
ax.set_title("Logistic Regression: Blacklist Probability vs ΔT1")
ax.set_xlim(0, 800); ax.legend(); ax.grid(alpha=0.3)
save(fig, "D_lr_prob_curve")

# 圖E：F1 vs threshold（細粒度搜尋）
fine_thresholds = np.arange(1, 500, 1)
fine_results = []
for t in fine_thresholds:
    pred = (df["dt1_days"] <= t).astype(int).values
    f1 = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    fine_results.append({"t": t, "f1": f1, "prec": prec, "rec": rec})
fine_df = pd.DataFrame(fine_results)
best_t_f1 = fine_df.loc[fine_df["f1"].idxmax(), "t"]

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FAFAFA")
ax.plot(fine_df["t"], fine_df["f1"],   color="#2CA02C", linewidth=2, label="F1")
ax.plot(fine_df["t"], fine_df["prec"], color=COLOR_B,   linewidth=1.5, alpha=0.8, label="Precision")
ax.plot(fine_df["t"], fine_df["rec"],  color=COLOR_N,   linewidth=1.5, alpha=0.8, label="Recall")
ax.axvline(best_t_f1, color="orange", linestyle="--", linewidth=2,
           label=f"Best F1 threshold = {best_t_f1}d (F1={fine_df.loc[fine_df['f1'].idxmax(),'f1']:.3f})")
ax.set_xlabel("Threshold (days)"); ax.set_ylabel("Score")
ax.set_title("Fine-Grained Threshold Search: F1 / Precision / Recall vs ΔT1 Threshold")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "E_fine_threshold_f1")

# ════════════════════════════════════════════════════════════════
# 4. 生成 MD 報告
# ════════════════════════════════════════════════════════════════
best_row = thresh_df.loc[thresh_df["F1"].idxmax()]
best_fine = fine_df.loc[fine_df["f1"].idxmax()]

md = f"""# ΔT1 分類實驗報告
> KYC Level2 完成 → 首次 TWD 入金 時間差（天）作為黑名單分類特徵

---

## 資料概況

| 項目 | 數值 |
|------|------|
| 總樣本數 | {len(df):,} |
| 黑名單數 | {int(y.sum()):,} |
| 黑名單比例 | {y.mean()*100:.2f}% |
| 正常用戶中位數 ΔT1 | {df[df['status']==0]['dt1_days'].median():.0f} 天 |
| 黑名單中位數 ΔT1 | {df[df['status']==1]['dt1_days'].median():.0f} 天 |

**核心觀察**：黑名單用戶完成 KYC 後的首次入金時間中位數僅 {df[df['status']==1]['dt1_days'].median():.0f} 天，
正常用戶為 {df[df['status']==0]['dt1_days'].median():.0f} 天，差距達 **{df[df['status']==0]['dt1_days'].median()/df[df['status']==1]['dt1_days'].median():.1f} 倍**。

---

## 方法一：規則式 Threshold 分類

> 規則：若 ΔT1 ≤ threshold，則標記為黑名單

| Threshold (天) | Precision | Recall | F1 | Specificity | Flagged % |
|:-:|:-:|:-:|:-:|:-:|:-:|
"""

for _, row in thresh_df.iterrows():
    marker = " ← **best F1**" if row["threshold_days"] == best_row["threshold_days"] else ""
    md += (f"| {int(row['threshold_days'])} | {row['Precision']:.3f} | {row['Recall']:.3f} | "
           f"{row['F1']:.3f} | {row['Specificity']:.3f} | {row['Flagged_pct']:.1f}%{marker} |\n")

md += f"""
### 最佳規則式 Threshold（F1 最大）

- **Threshold = {int(best_row['threshold_days'])} 天**
- Precision = {best_row['Precision']:.3f}（標記為黑名單中真正是黑名單的比例）
- Recall = {best_row['Recall']:.3f}（黑名單中被抓到的比例）
- F1 = {best_row['F1']:.3f}
- Specificity = {best_row['Specificity']:.3f}（正常用戶不被誤標的比例）
- 標記用戶比例 = {best_row['Flagged_pct']:.1f}%

### 細粒度搜尋（1天步長，1~499天）

- **最佳 F1 Threshold = {int(best_fine['t'])} 天**
- F1 = {best_fine['f1']:.4f}
- Precision = {best_fine['prec']:.4f}
- Recall = {best_fine['rec']:.4f}

---

## 方法二：Logistic Regression（5-fold Cross Validation）

> 以 ΔT1（天）為單一特徵，訓練 Logistic Regression，使用 class_weight='balanced' 處理類別不平衡

| 指標 | 數值 |
|------|------|
| AUC-ROC | **{auc_roc:.4f}** |
| Average Precision (AP) | **{ap:.4f}** |

### 最佳 Threshold 推導

LR 模型輸出 P(黑名單 | ΔT1)，可反推對應的 ΔT1 天數門檻：

| 方法 | Probability Threshold | 對應 ΔT1 |
|------|:---:|:---:|
| prob = 0.5 | 0.500 | **{dt1_50:.0f} 天** |
| Youden's J（最大化 TPR-FPR） | {best_thresh_youden:.3f} | **{dt1_youden:.0f} 天** |
| Best F1 | {best_thresh_f1:.3f} | **{dt1_f1:.0f} 天** |

---

## 不同 Threshold 的取捨分析

```
Threshold 越小（例如 7 天）：
  ✅ Precision 高 → 標記的人大多是真黑名單
  ❌ Recall 低  → 很多黑名單沒被抓到
  ❌ Flagged 少 → 覆蓋率不足

Threshold 越大（例如 365 天）：
  ✅ Recall 高  → 大部分黑名單都被抓到
  ❌ Precision 低 → 誤標很多正常用戶
  ❌ Flagged 多 → 人工審查成本高

建議操作點：
  - 高精準模式（人工複審資源有限）：threshold ≈ 7~14 天
  - 平衡模式（F1 最佳）：threshold ≈ {int(best_fine['t'])} 天
  - 高召回模式（不想漏掉任何黑名單）：threshold ≈ 90~180 天
```

---

## 圖表說明

| 檔案 | 說明 |
|------|------|
| A_threshold_metrics.png | 各 threshold 下的 Precision/Recall/F1/Specificity 曲線 |
| B_roc_curve.png | LR 的 ROC 曲線（AUC={auc_roc:.3f}） |
| C_pr_curve.png | LR 的 Precision-Recall 曲線（AP={ap:.3f}） |
| D_lr_prob_curve.png | LR 模型：P(黑名單) vs ΔT1 天數 |
| E_fine_threshold_f1.png | 細粒度 threshold 搜尋（最佳 F1 = {int(best_fine['t'])} 天） |

---

## 結論

1. **ΔT1 單特徵具有顯著區分力**，AUC-ROC = {auc_roc:.3f}，遠高於隨機（0.5）。
2. **規則式 threshold 簡單有效**，建議依業務需求選擇：
   - 嚴格模式：≤ 14 天
   - 平衡模式：≤ {int(best_fine['t'])} 天（F1 最佳）
3. **黑名單用戶的行為模式明確**：KYC 完成後極短時間內入金，符合「人頭戶準備好接錢」的假設。
4. 此特徵可作為風控規則引擎的一個獨立訊號，或與其他特徵組合進入 ML 模型。
"""

md_path = f"{OUT_DIR}/dt1_classification_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\nReport saved: {md_path}")
print("Done.")
