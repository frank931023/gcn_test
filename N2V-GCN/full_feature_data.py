import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. 讀取原始資料 (完全不讀 features.csv)
print("正在讀取原始資料...")
train_label = pd.read_csv('../data/train_label.csv')
predict_label = pd.read_csv('../data/predict_label.csv')
user_info = pd.read_csv('../data/user_info.csv')  # 代替 features_base
usdt_trading = pd.read_csv('../data/usdt_twd_trading.csv')
crypto_transfer = pd.read_csv('../data/crypto_transfer.csv')
twd_transfer = pd.read_csv('../data/twd_transfer.csv')
usdt_swap = pd.read_csv('../data/usdt_swap.csv')
degree_df = pd.read_csv('../data/edge/merged_degree_asymmetry.csv')
shared_ip = pd.read_csv('../data/edge/shared_ip_edges.csv')

# --- 強制型別轉換：確保所有 ID 都是 int，防止 428873 這種合併失敗 ---
id_tables = [train_label, predict_label, user_info, usdt_trading, crypto_transfer, twd_transfer, usdt_swap, degree_df]
for table in id_tables:
    table['user_id'] = table['user_id'].astype(int)

# 2. 建立全體名單 (訓練集 + 預測集)
all_users = pd.DataFrame(pd.concat([train_label['user_id'], predict_label['user_id']]).unique(), columns=['user_id'])

# === 3. USDT 交易特徵 ===
print("計算 USDT 特徵...")
usdt_trading['trade_amt'] = (usdt_trading['trade_samount'] / 1e8) * (usdt_trading['twd_srate'] / 1e8)
usdt_agg = usdt_trading.groupby('user_id').agg(
    usdt_trade_count=('id', 'count'),
    usdt_total_amount=('trade_amt', 'sum'),
    usdt_avg_amount=('trade_amt', 'mean'),
    usdt_max_amount=('trade_amt', 'max'),
    usdt_std_amount=('trade_amt', 'std'),
    usdt_buy_count=('is_buy', 'sum'),
    usdt_buy_ratio=('is_buy', 'mean'),
    usdt_market_ratio=('is_market', 'mean'),
    usdt_ip_count=('source_ip_hash', 'nunique'),
    usdt_source_avg=('source', 'mean')
).reset_index()

# === 4. 加密貨幣轉帳特徵 (含進階比例) ===
print("計算 Crypto 特徵與比例...")
crypto_transfer['created_at'] = pd.to_datetime(crypto_transfer['created_at'], errors='coerce')
crypto_transfer['is_night'] = crypto_transfer['created_at'].dt.hour.between(0, 6)

crypto_agg = crypto_transfer.groupby('user_id').agg(
    crypto_tx_count=('id', 'count'),
    crypto_total_amount=('ori_samount', lambda x: (pd.to_numeric(x, errors='coerce').fillna(0).astype(float)/1e8).sum()),
    crypto_avg_amount=('ori_samount', lambda x: (pd.to_numeric(x, errors='coerce').fillna(0).astype(float)/1e8).mean()),
    crypto_max_amount=('ori_samount', lambda x: (pd.to_numeric(x, errors='coerce').fillna(0).astype(float)/1e8).max()),
    crypto_deposit_count=('kind', lambda x: (x == 1).sum()),
    crypto_currency_count=('currency', 'nunique'),
    crypto_from_wallet_count=('from_wallet_hash', 'nunique'),
    crypto_to_wallet_count=('to_wallet_hash', 'nunique'),
    crypto_relation_user_count=('relation_user_id', 'nunique'),
    crypto_ip_count=('source_ip_hash', 'nunique'),
    night_tx_ratio=('is_night', 'mean'),
    internal_tx_ratio=('sub_kind', lambda x: (x == 1).mean())
).reset_index()

# 計算出入金不對稱比 (Out/In Ratio)
in_amt = crypto_transfer[crypto_transfer['kind'] == 0].groupby('user_id')['ori_samount'].sum() / 1e8
out_amt = crypto_transfer[crypto_transfer['kind'] == 1].groupby('user_id')['ori_samount'].sum() / 1e8
out_in_df = ((out_amt + 1) / (in_amt + 1)).fillna(0).rename('out_in_ratio').reset_index()

# === 5. 台幣轉帳特徵 ===
print("計算台幣特徵...")
twd_agg = twd_transfer.groupby('user_id').agg(
    twd_tx_count=('id', 'count'),
    twd_total_amount=('ori_samount', lambda x: (x/1e8).sum()),
    twd_avg_amount=('ori_samount', lambda x: (x/1e8).mean()),
    twd_max_amount=('ori_samount', lambda x: (x/1e8).max()),
    twd_deposit_count=('kind', lambda x: (x == 1).sum()),
    twd_ip_count=('source_ip_hash', 'nunique')
).reset_index()

# === 6. USDT 閃兌特徵 ===
print("計算閃兌特徵...")
swap_agg = usdt_swap.groupby('user_id').agg(
    swap_count=('id', 'count'),
    swap_twd_total=('twd_samount', 'sum'),
    swap_twd_avg=('twd_samount', 'mean'),
    swap_currency_total=('currency_samount', 'sum'),
    swap_currency_avg=('currency_samount', 'mean'),
    swap_kind_count=('kind', 'nunique')
).reset_index()

# === 7. IP 共用網路特徵 ===
print("計算 IP 共用特徵...")
ip_list = pd.concat([
    shared_ip[['user_id_1', 'source_ip_hash']].rename(columns={'user_id_1': 'user_id'}),
    shared_ip[['user_id_2', 'source_ip_hash']].rename(columns={'user_id_2': 'user_id'})
])
ip_agg = ip_list.groupby('user_id').agg(
    shared_ip_degree=('source_ip_hash', 'size'),
    shared_ip_unique_count=('source_ip_hash', 'nunique')
).reset_index()

# ---------------------------------------------------------
# 8. 最終合併 (以 all_users 為核心)
# ---------------------------------------------------------
print("正在執行全體合併...")
df = all_users.merge(user_info, on='user_id', how='left')
df = df.merge(usdt_agg, on='user_id', how='left')
df = df.merge(crypto_agg, on='user_id', how='left')
df = df.merge(out_in_df, on='user_id', how='left')
df = df.merge(twd_agg, on='user_id', how='left')
df = df.merge(swap_agg, on='user_id', how='left')
df = df.merge(ip_agg, on='user_id', how='left')
df = df.merge(degree_df[['user_id', 'degree_diff']], on='user_id', how='left')

# 補上標籤 (只從 train_label 補，predict_label 的人會是 NaN)
df = df.merge(train_label[['user_id', 'status']], on='user_id', how='left')

# 9. 缺失值處理
print("處理缺失值...")
# 數值類補 0
num_cols = df.select_dtypes(include=[np.number]).columns
num_cols = [c for c in num_cols if c != 'status']
df[num_cols] = df[num_cols].fillna(0)

# 類別類補 'Unknown'
cat_cols = ['sex', 'career', 'income_source', 'user_source']
df[cat_cols] = df[cat_cols].fillna('Unknown')

# 10. 儲存與驗證
df.to_csv('all_features_analysis.csv', index=False)
print(f"\n✅ 成功！全體特徵表已生成。")
print(f"總人數: {len(df)} (訓練: {len(train_label)}, 預測: {len(predict_label)})")
print(f"總欄位數: {len(df.columns)}")
