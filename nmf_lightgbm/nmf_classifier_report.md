# NMF + LightGBM 詐騙偵測分類器實驗報告

## 方法介紹

本實驗採用 **NMF（Non-negative Matrix Factorization，非負矩陣分解）+ LightGBM** 的兩階段混合方法，用於偵測加密貨幣交易平台的詐騙用戶。

### 核心思想

詐騙農場通常會利用同批設備或 IP 註冊大量人頭戶（女巫攻擊 Sybil Attack），這些帳戶的 IP 使用模式高度相似。透過矩陣分解技術，可以將高維稀疏的 User × IP 矩陣壓縮成低維的「IP 使用模式向量」，捕捉隱藏的群體特徵，再結合交易行為特徵進行分類。

---

## 方法原理

### 1. NMF 矩陣分解

**輸入矩陣**：User × IP 二元矩陣（37655 用戶 × 471 IP）
- 矩陣元素：1 = 該用戶使用過該 IP，0 = 未使用
- 稀疏度：99.78%（大部分元素為 0）

**分解目標**：
```
V ≈ W × H
```
- V：原始 User × IP 矩陣（37655 × 471）
- W：用戶特徵矩陣（37655 × 20）← 每個用戶的 IP 使用模式向量
- H：IP 群體矩陣（20 × 471）← 20 個潛在 IP 群體

**為什麼用 NMF？**
- 非負約束：符合「使用次數」的物理意義（不會有負數）
- 可解釋性：分解出的群體特徵有明確意義（例如：群體 0 = 詐騙農場 A 的 IP 集合）
- 降維效果：471 維 → 20 維，大幅減少特徵數量

**分解結果**：
- 分解維度：20（代表 20 個潛在的 IP 使用群體）
- 重建誤差：38.94（越低越好，表示分解後能還原原矩陣的程度）

### 2. 特徵工程

除了 NMF 提取的 20 個群體特徵，還加入以下交易行為特徵：

**資金流特徵**：
- `twd_in_sum`：法幣入金總額
- `twd_in_mean`：法幣入金平均金額
- `twd_in_count`：法幣入金次數
- `crypto_out_sum`：加密幣出金總額
- `crypto_out_mean`：加密幣出金平均金額
- `crypto_out_count`：加密幣出金次數

**交易特徵**：
- `trade_sum`：USDT 交易總量
- `trade_mean`：USDT 交易平均金額
- `trade_count`：交易次數

**IP 共用特徵**：
- `unique_ip_count`：該用戶使用過的獨立 IP 數量
- `total_ip_records`：該用戶的 IP 紀錄總數

**衍生特徵**：
- `conversion_rate`：資金轉換率 = 加密幣出金 / 法幣入金（詐騙用戶傾向 100% 轉出）

**最終特徵數**：32 個（20 個 NMF + 12 個交易行為特徵）

### 3. LightGBM 分類

**模型參數**：
- 目標函數：binary（二元分類）
- 評估指標：AUC（ROC 曲線下面積）
- 樹深度：31 葉節點
- 學習率：0.05
- 特徵採樣：80%
- 樣本採樣：80%
- 類別權重：自動平衡（正常:黑名單 ≈ 30:1）

**訓練策略**：
- 訓練集 / 測試集：70% / 30%（分層抽樣保持黑名單比例）
- Early Stopping：驗證集 AUC 連續 20 輪未提升則停止
- 最佳迭代次數：15 輪

---

## 實驗結果

### 資料統計

| 項目 | 數量 |
|------|------|
| 總用戶數 | 51,017 |
| 黑名單用戶 | 1,640（3.2%）|
| 正常用戶 | 49,377（96.8%）|
| 有 IP 紀錄的用戶 | 37,655 |
| 共用 IP 數（≥3人）| 471 |
| 矩陣稀疏度 | 99.78% |

### 模型表現

**整體指標**：
- **ROC-AUC：0.8263**（0.8 以上為良好）
- **準確率：95%**

**分類報告**：

| 類別 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| 正常用戶 | 0.98 | 0.97 | 0.97 | 10,926 |
| 黑名單用戶 | 0.26 | 0.32 | 0.29 | 371 |

**混淆矩陣**：

|  | 預測正常 | 預測黑名單 |
|--|----------|------------|
| **實際正常** | 10,578 | 348 |
| **實際黑名單** | 251 | 120 |

**關鍵指標解讀**：
- 黑名單召回率（Recall）：32%
  - 成功偵測到 120 / 371 個黑名單用戶
  - 漏報率 68%（251 個黑名單被誤判為正常）
- 正常用戶準確率（Precision）：98%
  - 誤判率低，只有 348 個正常用戶被標記為黑名單
- 黑名單精確率（Precision）：26%
  - 被標記為黑名單的用戶中，只有 26% 真的是黑名單
  - 表示模型較保守，寧可漏報也不誤殺

### 特徵重要性分析

**Top 15 重要特徵**：

| 排名 | 特徵名稱 | 重要性（Gain）| 類型 |
|------|----------|---------------|------|
| 1 | crypto_out_sum | 82,859 | 交易行為 |
| 2 | twd_in_sum | 26,103 | 交易行為 |
| 3 | twd_in_mean | 25,432 | 交易行為 |
| 4 | trade_mean | 21,165 | 交易行為 |
| 5 | crypto_out_mean | 13,983 | 交易行為 |
| 6 | conversion_rate | 13,190 | 衍生特徵 |
| 7 | trade_sum | 9,152 | 交易行為 |
| 8 | trade_count | 5,853 | 交易行為 |
| 9 | crypto_out_count | 2,465 | 交易行為 |
| 10 | **nmf_0** | **2,144** | **NMF 群體特徵** |
| 11 | twd_in_count | 1,904 | 交易行為 |
| 12 | unique_ip_count | 1,604 | IP 特徵 |
| 13 | **nmf_8** | **545** | **NMF 群體特徵** |
| 14 | **nmf_5** | **261** | **NMF 群體特徵** |
| 15 | nmf_1 | 0 | NMF 群體特徵 |

**關鍵發現**：
1. **交易金額特徵主導**：`crypto_out_sum`（加密幣出金總額）重要性遠超其他特徵，說明詐騙用戶的出金行為是最強信號
2. **NMF 特徵有效**：`nmf_0`、`nmf_8`、`nmf_5` 進入 Top 15，證明 IP 使用模式群體特徵確實有助於分類
3. **轉換率關鍵**：`conversion_rate` 排名第 6，符合「詐騙用戶傾向 100% 轉出」的假設
4. **IP 共用特徵貢獻有限**：`unique_ip_count` 排名第 12，單純的 IP 數量不如群體模式重要

---

## 程式碼說明

### 主要流程

```python
# 1. 載入資料
twd_transfer, crypto, trading, train_label = load_data()

# 2. 建立 User × IP 矩陣
ip_records = concat([twd, crypto, trading])
user_ip_matrix = pivot_table(ip_records)  # 37655 × 471

# 3. NMF 矩陣分解
nmf = NMF(n_components=20)
user_features_nmf = nmf.fit_transform(user_ip_matrix)  # 37655 × 20

# 4. 提取其他特徵
features = merge([twd_stats, crypto_stats, trade_stats, ip_stats, nmf_features])

# 5. 訓練 LightGBM
X_train, X_test, y_train, y_test = train_test_split(features, labels)
model = lgb.train(params, train_data)

# 6. 評估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 關鍵參數

**NMF 分解**：
```python
nmf = NMF(
    n_components=20,        # 分解成 20 個群體
    init='nndsvd',          # 初始化方法（適合稀疏矩陣）
    random_state=42,        # 隨機種子
    max_iter=500            # 最大迭代次數
)
```

**LightGBM 訓練**：
```python
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "scale_pos_weight": 30  # 處理類別不平衡（正常:黑名單 = 30:1）
}
```

---

## 結論與建議

### 優點

1. **NMF 成功提取群體特徵**：`nmf_0` 進入 Top 10，證明矩陣分解能捕捉詐騙農場的 IP 共用模式
2. **ROC-AUC 0.83**：在高度不平衡資料（黑名單僅 3.2%）下表現良好
3. **誤判率低**：正常用戶準確率 98%，不會大量誤殺

### 限制

1. **黑名單召回率偏低（32%）**：漏報率 68%，可能因為：
   - 黑名單樣本太少（1640 人），模型學習不足
   - 部分詐騙用戶行為與正常用戶相似，難以區分
   - 模型參數偏保守（`scale_pos_weight` 可調整）

2. **交易金額特徵主導**：NMF 特徵重要性相對較低，IP 模式可能不是最強信號

### 改進方向

1. **增加 NMF 維度**：從 20 提升到 50，捕捉更細緻的群體模式
2. **調整類別權重**：降低 `scale_pos_weight`，提高黑名單召回率（但會增加誤判）
3. **結合 GCN**：用圖神經網路直接學習 IP 共現圖結構，可能比 NMF 更有效
4. **時序特徵**：加入 24 小時行為模式（API 下單時段、棋盤狀熱力圖特徵）
5. **集成學習**：結合 NMF + GCN + 規則引擎，多模型投票

---

## 輸出檔案

- `nmf_classifier.py`：完整程式碼
- `nmf_feature_importance.png`：特徵重要性視覺化圖表
- `nmf_classifier_report.md`：本報告

---

## 參考資料

- NMF 原理：Lee & Seung (1999), "Learning the parts of objects by non-negative matrix factorization"
- LightGBM 文件：https://lightgbm.readthedocs.io/
- 女巫攻擊偵測：Viswanath et al. (2010), "An Analysis of Social Network-Based Sybil Defenses"
