import pandas as pd

path = "predict_result.csv"
df = pd.read_csv(path)
print(df.iloc[:, 1])  # 顯示第二欄全部內容
counts = df.iloc[:, 1].value_counts()
ratio = counts.get(1, 0) / counts.sum() if counts.sum() > 0 else 0
print(f"0 的数量: {counts.get(0, 0)}")
print(f"1 的数量: {counts.get(1, 0)}")
print(f"1 占比: {ratio:.4f}")
