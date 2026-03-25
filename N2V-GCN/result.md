# v8 + all_features_analysis.csv 預測與分析流程

## 1. 預測流程（end_model.py）

- 使用 v8 GAT 結構，特徵來源為 all_features_analysis.csv。
- 預測名單以 predict_label.csv 為主，確保所有 user_id 都有預測。
- 輸出 predict_result.csv，保證 12,753 筆。
- 動態門檻：強制抓出機率最高的前 500 名為 1，其餘為 0。

## 2. 分析流程（analysis_result.py）

- 讀取 predict_result.csv 與 all_features_analysis.csv。
- 若 all_features_analysis.csv 有 status 欄，會自動對齊真實標籤。
- 輸出：
  - 預測標籤分布圖（predict_label_distribution.png）
  - 若有真實標籤，輸出混淆矩陣圖（confusion_matrix.png）與分類報告。

---

## 使用說明

1. 執行 end_model.py 產生 predict_result.csv。
2. 執行 analysis_result.py 產生圖表與統計。
3. 查看 predict_label_distribution.png、confusion_matrix.png（如有標籤）。
