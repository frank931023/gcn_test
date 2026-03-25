import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 讀取你的預測結果（包含機率的那個檔案）
# 如果你之前的程式碼有存 pred_probs，請讀取它
# 這裡假設你有一個 DataFrame 叫 df_result，裡面有 'pred_prob' 欄位
try:
    df = pd.read_csv('predict_result_with_probs.csv') # 建議在你產出結果時順便存機率
    probs = df['pred_prob'].values
except:
    # 如果沒存，請在你的模型預測完後直接跑這段繪圖邏輯
    print("請確保你有 predict_result_with_probs.csv 檔案")
    probs = np.random.beta(1, 20, 12753) # 模擬資料：大部分機率都很低
# 2. 設定門檻
best_t = 0.35  # 假設的固定門檻 (10人那個)
dynamic_t = np.sort(probs)[-510] # 動態門檻 (510人那個)

# 3. 繪圖
plt.figure(figsize=(12, 6))
sns.histplot(probs, bins=100, kde=True, color='skyblue', label='Predict Probabilities')

# 標示兩條門檻線
plt.axvline(best_t, color='red', linestyle='--', label=f'Fixed Threshold (Strict, n=10): {best_t:.3f}')
plt.axvline(dynamic_t, color='green', linestyle='-', label=f'Dynamic Threshold (Top 4%, n=510): {dynamic_t:.3f}')

plt.title('Distribution of Predicted Probabilities (12,753 Users)', fontsize=15)
plt.xlabel('Probability of being Blacklist', fontsize=12)
plt.ylabel('User Count', fontsize=12)
plt.yscale('log') # 使用對數坐標，否則 0 附近的人數會壓過所有人
plt.legend()
plt.grid(alpha=0.3)

plt.savefig('prob_distribution.png')
print(f"圖表已儲存！動態門檻值為: {dynamic_t:.4f}")
plt.show()