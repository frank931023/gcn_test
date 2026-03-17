from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. 讀取行為特徵 (你之前算好的 features.csv 或擴展表)
behavior_df = pd.read_csv('extended_features_analysis.csv')
n2v_df = pd.read_csv("user_n2v.embeddings", sep=' ', skiprows=1, header=None)
n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]

# 合併特徵
final_feature_df = behavior_df.merge(n2v_df, on='user_id', how='inner')
X = final_feature_df.drop(columns=['user_id', 'status'])
X = X.select_dtypes(include=['number'])

y = final_feature_df['status']
y = pd.to_numeric(y, errors='coerce')


# 2. 複合採樣策略
# 先用 Near-Miss 減少過多且無代表性的正常樣本
nm = NearMiss(version=1, sampling_strategy=0.1) 
X_res, y_res = nm.fit_resample(X, y)

# 再用 SMOTE 針對黑名單進行上採樣，達到 1:1 平衡
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_final, y_final = smote.fit_resample(X_res, y_res)

print(f"採樣後資料筆數：{len(y_final)} (平衡狀態)")