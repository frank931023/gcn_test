import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('visulize_result/outputs_uturn', exist_ok=True)

# 讀取數據
print("讀取數據...")
twd_transfer = pd.read_csv('data/twd_transfer.csv')
usdt_trading = pd.read_csv('data/usdt_twd_trading.csv')
crypto_transfer = pd.read_csv('data/crypto_transfer.csv')
train_label = pd.read_csv('data/train_label.csv')

# 轉換時間欄位
twd_transfer['created_at'] = pd.to_datetime(twd_transfer['created_at'])
usdt_trading['updated_at'] = pd.to_datetime(usdt_trading['updated_at'])
crypto_transfer['created_at'] = pd.to_datetime(crypto_transfer['created_at'])

print(f"TWD Transfer: {len(twd_transfer)} 筆")
print(f"USDT Trading: {len(usdt_trading)} 筆")
print(f"Crypto Transfer: {len(crypto_transfer)} 筆")

# 建立用戶標籤字典
user_labels = dict(zip(train_label['user_id'], train_label['status']))

def calculate_balance_timeline(user_id):
    """計算用戶的餘額時間線"""
    user_twd = twd_transfer[twd_transfer['user_id'] == user_id].copy()
    user_usdt = usdt_trading[usdt_trading['user_id'] == user_id].copy()
    user_crypto = crypto_transfer[crypto_transfer['user_id'] == user_id].copy()
    
    events = []
    
    # TWD 加值/提領
    for _, row in user_twd.iterrows():
        amount = row['ori_samount'] / 1e8
        if row['kind'] == 0:
            events.append({'time': row['created_at'], 'type': 'twd_deposit', 
                          'twd_change': amount, 'usdt_change': 0})
        else:
            events.append({'time': row['created_at'], 'type': 'twd_withdraw', 
                          'twd_change': -amount, 'usdt_change': 0})
    
    # USDT 交易
    for _, row in user_usdt.iterrows():
        usdt_amount = row['trade_samount'] / 1e8
        twd_rate = row['twd_srate'] / 1e8
        twd_amount = usdt_amount * twd_rate
        
        if row['is_buy'] == 1:
            events.append({'time': row['updated_at'], 'type': 'buy_usdt',
                          'twd_change': -twd_amount, 'usdt_change': usdt_amount})
        else:
            events.append({'time': row['updated_at'], 'type': 'sell_usdt',
                          'twd_change': twd_amount, 'usdt_change': -usdt_amount})
    
    # Crypto 提領
    for _, row in user_crypto.iterrows():
        if row['kind'] == 1:
            crypto_amount = row['ori_samount'] / 1e8
            twd_rate = row['twd_srate'] / 1e8 if pd.notna(row['twd_srate']) else 30
            twd_value = crypto_amount * twd_rate
            events.append({'time': row['created_at'], 'type': 'crypto_withdraw',
                          'twd_change': 0, 'usdt_change': -twd_value / 30})
    
    if not events:
        return None
    
    events_df = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
    events_df['twd_balance'] = events_df['twd_change'].cumsum()
    events_df['usdt_balance'] = events_df['usdt_change'].cumsum()
    events_df['total_balance_twd'] = events_df['twd_balance'] + events_df['usdt_balance'] * 30
    
    return events_df

def extract_uturn_features(events_df):
    """提取 U 型迴轉特徵"""
    if events_df is None or len(events_df) == 0:
        return {'zero_balance_count': 0, 'avg_balance_duration_hours': 0,
                'max_balance': 0, 'balance_volatility': 0, 'uturn_ratio': 0, 'pulse_count': 0}
    
    zero_threshold = 1000
    zero_balance_count = (events_df['total_balance_twd'] < zero_threshold).sum()
    
    balance_durations = []
    current_duration = timedelta(0)
    last_time = None
    
    for idx, row in events_df.iterrows():
        if row['total_balance_twd'] > zero_threshold:
            if last_time is not None:
                current_duration += row['time'] - last_time
            last_time = row['time']
        else:
            if current_duration.total_seconds() > 0:
                balance_durations.append(current_duration.total_seconds() / 3600)
                current_duration = timedelta(0)
            last_time = row['time']
    
    avg_balance_duration = np.mean(balance_durations) if balance_durations else 0
    max_balance = events_df['total_balance_twd'].max()
    balance_volatility = events_df['total_balance_twd'].std() if len(events_df) > 1 else 0
    
    large_transactions = events_df[abs(events_df['twd_change'] + events_df['usdt_change'] * 30) > 10000]
    uturn_count = 0
    
    for idx in large_transactions.index:
        next_events = events_df[events_df['time'] > events_df.loc[idx, 'time']]
        next_events = next_events[next_events['time'] <= events_df.loc[idx, 'time'] + timedelta(hours=24)]
        if len(next_events) > 0 and next_events['total_balance_twd'].min() < zero_threshold:
            uturn_count += 1
    
    uturn_ratio = uturn_count / len(events_df) if len(events_df) > 0 else 0
    
    pulse_count = 0
    for i in range(1, len(events_df) - 1):
        prev_balance = events_df.iloc[i-1]['total_balance_twd']
        curr_balance = events_df.iloc[i]['total_balance_twd']
        next_balance = events_df.iloc[i+1]['total_balance_twd']
        if prev_balance < zero_threshold and curr_balance > zero_threshold * 10 and next_balance < zero_threshold:
            pulse_count += 1
    
    return {'zero_balance_count': zero_balance_count, 'avg_balance_duration_hours': avg_balance_duration,
            'max_balance': max_balance, 'balance_volatility': balance_volatility,
            'uturn_ratio': uturn_ratio, 'pulse_count': pulse_count}

# 提取特徵
print("\n提取 U 型迴轉特徵...")
all_users = set(twd_transfer['user_id']) | set(usdt_trading['user_id']) | set(crypto_transfer['user_id'])
print(f"總共 {len(all_users)} 個用戶")

features_list = []
for user_id in list(all_users)[:1000]:
    events_df = calculate_balance_timeline(user_id)
    features = extract_uturn_features(events_df)
    features['user_id'] = user_id
    features['label'] = user_labels.get(user_id, 0)
    features_list.append(features)

features_df = pd.DataFrame(features_list)
features_df.to_csv('visulize_result/outputs_uturn/uturn_features.csv', index=False)
print(f"\n特徵已保存: {len(features_df)} 個用戶")

# 統計分析
print("\n=== 特徵統計 ===")
print("\n正常用戶 (label=0):")
print(features_df[features_df['label'] == 0].describe())
print("\n黑名單用戶 (label=1):")
print(features_df[features_df['label'] == 1].describe())

# 視覺化
features_to_plot = [
    ('zero_balance_count', '餘額歸零次數'),
    ('avg_balance_duration_hours', '平均餘額停留時間 (小時)'),
    ('max_balance', '最大餘額 (TWD)'),
    ('balance_volatility', '餘額波動率'),
    ('uturn_ratio', 'U 型迴轉比率'),
    ('pulse_count', '脈衝次數')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('U 型迴轉特徵分佈對比', fontsize=16, fontweight='bold')

for idx, (feature, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    normal_data = features_df[features_df['label'] == 0][feature]
    blacklist_data = features_df[features_df['label'] == 1][feature]
    ax.hist(normal_data, bins=30, alpha=0.6, label='正常用戶', color='blue', edgecolor='black')
    ax.hist(blacklist_data, bins=30, alpha=0.6, label='黑名單用戶', color='red', edgecolor='black')
    ax.set_xlabel(title)
    ax.set_ylabel('用戶數量')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_distribution.png', dpi=300, bbox_inches='tight')
print("\n特徵分佈圖已保存")

# 散布圖對比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('U 型迴轉特徵散布圖對比', fontsize=16, fontweight='bold')

for idx, (feature, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    normal_data = features_df[features_df['label'] == 0][feature]
    blacklist_data = features_df[features_df['label'] == 1][feature]
    
    # 添加一些隨機抖動以避免點重疊
    normal_x = np.random.normal(0, 0.04, size=len(normal_data))
    blacklist_x = np.random.normal(1, 0.04, size=len(blacklist_data))
    
    ax.scatter(normal_x, normal_data, alpha=0.5, s=50, color='blue', 
               label=f'正常用戶 (n={len(normal_data)})', edgecolors='black', linewidths=0.5)
    ax.scatter(blacklist_x, blacklist_data, alpha=0.7, s=100, color='red', 
               label=f'黑名單用戶 (n={len(blacklist_data)})', edgecolors='black', linewidths=1)
    
    # 添加平均線
    ax.axhline(y=normal_data.mean(), color='blue', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'正常平均: {normal_data.mean():.2f}')
    ax.axhline(y=blacklist_data.mean(), color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'黑名單平均: {blacklist_data.mean():.2f}')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['正常用戶', '黑名單用戶'])
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_scatter.png', dpi=300, bbox_inches='tight')
print("散布圖已保存")

# 小提琴圖對比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('U 型迴轉特徵小提琴圖對比', fontsize=16, fontweight='bold')

for idx, (feature, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    # 準備數據
    plot_data = pd.DataFrame({
        'value': pd.concat([features_df[features_df['label'] == 0][feature],
                           features_df[features_df['label'] == 1][feature]]),
        'label': ['正常用戶'] * len(features_df[features_df['label'] == 0]) + 
                 ['黑名單用戶'] * len(features_df[features_df['label'] == 1])
    })
    
    # 繪製小提琴圖
    parts = ax.violinplot([features_df[features_df['label'] == 0][feature],
                           features_df[features_df['label'] == 1][feature]],
                          positions=[0, 1], showmeans=True, showmedians=True)
    
    # 設置顏色
    for pc, color in zip(parts['bodies'], ['lightblue', 'lightcoral']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['正常用戶', '黑名單用戶'])
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_violin.png', dpi=300, bbox_inches='tight')
print("小提琴圖已保存")

# 條形圖對比（平均值 + 標準差）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('U 型迴轉特徵條形圖對比（平均值 ± 標準差）', fontsize=16, fontweight='bold')

for idx, (feature, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    normal_data = features_df[features_df['label'] == 0][feature]
    blacklist_data = features_df[features_df['label'] == 1][feature]
    
    means = [normal_data.mean(), blacklist_data.mean()]
    stds = [normal_data.std(), blacklist_data.std()]
    
    x_pos = [0, 1]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    # 在條形上方顯示數值
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std, f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['正常用戶', '黑名單用戶'])
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_barplot.png', dpi=300, bbox_inches='tight')
print("條形圖已保存")

# 雷達圖對比（標準化後的特徵）
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 標準化特徵值（0-1 範圍）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

normal_means = []
blacklist_means = []

for feature, _ in features_to_plot:
    normal_data = features_df[features_df['label'] == 0][feature]
    blacklist_data = features_df[features_df['label'] == 1][feature]
    
    # 合併數據進行標準化
    all_data = pd.concat([normal_data, blacklist_data]).values.reshape(-1, 1)
    scaler.fit(all_data)
    
    normal_scaled = scaler.transform(normal_data.values.reshape(-1, 1)).mean()
    blacklist_scaled = scaler.transform(blacklist_data.values.reshape(-1, 1)).mean()
    
    normal_means.append(normal_scaled)
    blacklist_means.append(blacklist_scaled)

# 設置角度
angles = np.linspace(0, 2 * np.pi, len(features_to_plot), endpoint=False).tolist()
normal_means += normal_means[:1]
blacklist_means += blacklist_means[:1]
angles += angles[:1]

# 繪製雷達圖
ax.plot(angles, normal_means, 'o-', linewidth=2, label='正常用戶', color='blue')
ax.fill(angles, normal_means, alpha=0.25, color='blue')
ax.plot(angles, blacklist_means, 'o-', linewidth=2, label='黑名單用戶', color='red')
ax.fill(angles, blacklist_means, alpha=0.25, color='red')

# 設置標籤
ax.set_xticks(angles[:-1])
ax.set_xticklabels([title for _, title in features_to_plot], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('特徵雷達圖對比（標準化後）', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_radar.png', dpi=300, bbox_inches='tight')
print("雷達圖已保存")

# 餘額階梯圖
print("\n繪製典型用戶的餘額階梯圖...")
normal_users = features_df[features_df['label'] == 0].nlargest(3, 'max_balance')['user_id'].tolist()
blacklist_users = features_df[features_df['label'] == 1].nlargest(3, 'pulse_count')['user_id'].tolist()

def plot_balance_step_chart(user_id, label_name, color, save_path):
    events_df = calculate_balance_timeline(user_id)
    if events_df is None or len(events_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.step(events_df['time'], events_df['total_balance_twd'], where='post', 
            linewidth=2, color=color, label='總資產價值 (TWD)')
    
    for event_type, marker, marker_color, marker_label in [
        ('twd_deposit', '^', 'green', 'TWD 加值'),
        ('twd_withdraw', 'v', 'red', 'TWD 提領'),
        ('buy_usdt', 'o', 'blue', '買入 USDT'),
        ('sell_usdt', 's', 'orange', '賣出 USDT'),
        ('crypto_withdraw', 'x', 'purple', 'Crypto 提領')
    ]:
        event_data = events_df[events_df['type'] == event_type]
        if len(event_data) > 0:
            ax.scatter(event_data['time'], event_data['total_balance_twd'], 
                      marker=marker, s=100, color=marker_color, label=marker_label, 
                      alpha=0.7, edgecolors='black', linewidths=1)
    
    ax.axhline(y=1000, color='red', linestyle='--', linewidth=1.5, 
               label='零餘額閾值 (1000 TWD)', alpha=0.7)
    ax.set_xlabel('時間', fontsize=12)
    ax.set_ylabel('帳戶總資產價值 (TWD)', fontsize=12)
    ax.set_title(f'用戶 {user_id} 的餘額階梯圖 ({label_name})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

for idx, user_id in enumerate(normal_users[:3]):
    plot_balance_step_chart(user_id, '正常用戶', 'blue',
                           f'visulize_result/outputs_uturn/balance_step_normal_{idx+1}.png')

for idx, user_id in enumerate(blacklist_users[:3]):
    plot_balance_step_chart(user_id, '黑名單用戶', 'red',
                           f'visulize_result/outputs_uturn/balance_step_blacklist_{idx+1}.png')

# 對比圖
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('餘額階梯圖對比：正常用戶 vs 黑名單用戶', fontsize=16, fontweight='bold')

if len(normal_users) > 0:
    events_df = calculate_balance_timeline(normal_users[0])
    if events_df is not None and len(events_df) > 0:
        axes[0].step(events_df['time'], events_df['total_balance_twd'], 
                    where='post', linewidth=2, color='blue')
        axes[0].axhline(y=1000, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0].set_ylabel('總資產價值 (TWD)', fontsize=12)
        axes[0].set_title(f'正常用戶 (User {normal_users[0]}) - 平緩波段', fontsize=12)
        axes[0].grid(True, alpha=0.3)

if len(blacklist_users) > 0:
    events_df = calculate_balance_timeline(blacklist_users[0])
    if events_df is not None and len(events_df) > 0:
        axes[1].step(events_df['time'], events_df['total_balance_twd'], 
                    where='post', linewidth=2, color='red')
        axes[1].axhline(y=1000, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1].set_xlabel('時間', fontsize=12)
        axes[1].set_ylabel('總資產價值 (TWD)', fontsize=12)
        axes[1].set_title(f'黑名單用戶 (User {blacklist_users[0]}) - 脈衝波形', fontsize=12)
        axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/balance_comparison.png', dpi=300, bbox_inches='tight')
print("對比圖已保存")

# 特徵差異分析
print("\n=== 特徵差異分析 ===")
for feature, title in features_to_plot:
    normal_mean = features_df[features_df['label'] == 0][feature].mean()
    blacklist_mean = features_df[features_df['label'] == 1][feature].mean()
    diff_ratio = (blacklist_mean - normal_mean) / (normal_mean + 1e-10) * 100
    print(f"\n{title}:")
    print(f"  正常用戶平均值: {normal_mean:.2f}")
    print(f"  黑名單用戶平均值: {blacklist_mean:.2f}")
    print(f"  差異比率: {diff_ratio:+.2f}%")

print("\n所有視覺化已完成！輸出目錄: visulize_result/outputs_uturn/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('visulize_result/outputs_uturn', exist_ok=True)

# 讀取數據
print("讀取數據...")
twd_transfer = pd.read_csv('data/twd_transfer.csv')
usdt_trading = pd.read_csv('data/usdt_twd_trading.csv')
crypto_transfer = pd.read_csv('data/crypto_transfer.csv')
train_label = pd.read_csv('data/train_label.csv')

# 轉換時間欄位
twd_transfer['created_at'] = pd.to_datetime(twd_transfer['created_at'])
usdt_trading['updated_at'] = pd.to_datetime(usdt_trading['updated_at'])
crypto_transfer['created_at'] = pd.to_datetime(crypto_transfer['created_at'])

print(f"TWD Transfer: {len(twd_transfer)} 筆")
print(f"USDT Trading: {len(usdt_trading)} 筆")
print(f"Crypto Transfer: {len(crypto_transfer)} 筆")

# 建立用戶標籤字典
user_labels = dict(zip(train_label['user_id'], train_label['status']))

def calculate_balance_timeline(user_id):
    """
    計算用戶的餘額時間線
    模擬 TWD 和 USDT 的餘額變化
    """
    # 獲取該用戶的所有交易
    user_twd = twd_transfer[twd_transfer['user_id'] == user_id].copy()
    user_usdt = usdt_trading[usdt_trading['user_id'] == user_id].copy()
    user_crypto = crypto_transfer[crypto_transfer['user_id'] == user_id].copy()
    
    # 創建統一的交易事件列表
    events = []
    
    # TWD 加值/提領 (kind=0 是加值, kind=1 是提領)
    for _, row in user_twd.iterrows():
        amount = row['ori_samount'] / 1e8  # 轉換為實際金額
        if row['kind'] == 0:  # 加值
            events.append({
                'time': row['created_at'],
                'type': 'twd_deposit',
                'amount': amount,
                'twd_change': amount,
                'usdt_change': 0
            })
        else:  # 提領
            events.append({
                'time': row['created_at'],
                'type': 'twd_withdraw',
                'amount': amount,
                'twd_change': -amount,
                'usdt_change': 0
            })
    
    # USDT 交易 (is_buy=1 是買入USDT, is_buy=0 是賣出USDT)
    for _, row in user_usdt.iterrows():
        usdt_amount = row['trade_samount'] / 1e8
        twd_rate = row['twd_srate'] / 1e8
        twd_amount = usdt_amount * twd_rate
        
        if row['is_buy'] == 1:  # 買入 USDT (TWD -> USDT)
            events.append({
                'time': row['updated_at'],
                'type': 'buy_usdt',
                'amount': usdt_amount,
                'twd_change': -twd_amount,
                'usdt_change': usdt_amount
            })
        else:  # 賣出 USDT (USDT -> TWD)
            events.append({
                'time': row['updated_at'],
                'type': 'sell_usdt',
                'amount': usdt_amount,
                'twd_change': twd_amount,
                'usdt_change': -usdt_amount
            })
    
    # Crypto 提領 (kind=1 是提領)
    for _, row in user_crypto.iterrows():
        if row['kind'] == 1:  # 提領
            crypto_amount = row['ori_samount'] / 1e8
            twd_rate = row['twd_srate'] / 1e8 if pd.notna(row['twd_srate']) else 30
            twd_value = crypto_amount * twd_rate
            
            events.append({
                'time': row['created_at'],
                'type': 'crypto_withdraw',
                'amount': crypto_amount,
                'twd_change': 0,
                'usdt_change': -twd_value / 30  # 假設 USDT = 30 TWD
            })
    
    if not events:
        return None
    
    # 按時間排序
    events_df = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
    
    # 計算累積餘額
    events_df['twd_balance'] = events_df['twd_change'].cumsum()
    events_df['usdt_balance'] = events_df['usdt_change'].cumsum()
    events_df['total_balance_twd'] = events_df['twd_balance'] + events_df['usdt_balance'] * 30
    
    return events_df

def extract_uturn_features(events_df):
    """
    提取 U 型迴轉特徵
    """
    if events_df is None or len(events_df) == 0:
        return {
            'zero_balance_count': 0,
            'avg_balance_duration_hours': 0,
            'max_balance': 0,
            'balance_volatility': 0,
            'uturn_ratio': 0,
            'pulse_count': 0
        }
    
    # 1. 餘額歸零次數 (餘額 < 1000 TWD 視為歸零)
    zero_threshold = 1000
    zero_balance_count = (events_df['total_balance_twd'] < zero_threshold).sum()
    
    # 2. 資金停留時間 (計算每次餘額 > 0 的持續時間)
    balance_durations = []
    current_duration = timedelta(0)
    last_time = None
    
    for idx, row in events_df.iterrows():
        if row['total_balance_twd'] > zero_threshold:
            if last_time is not None:
                current_duration += row['time'] - last_time
            last_time = row['time']
        else:
            if current_duration.total_seconds() > 0:
                balance_durations.append(current_duration.total_seconds() / 3600)  # 轉換為小時
                current_duration = timedelta(0)
            last_time = row['time']
    
    avg_balance_duration = np.mean(balance_durations) if balance_durations else 0
    
    # 3. 最大餘額
    max_balance = events_df['total_balance_twd'].max()
    
    # 4. 餘額波動率
    balance_volatility = events_df['total_balance_twd'].std() if len(events_df) > 1 else 0
    
    # 5. U 型迴轉比率 (大額進出後歸零的次數 / 總交易次數)
    large_transactions = events_df[abs(events_df['twd_change'] + events_df['usdt_change'] * 30) > 10000]
    uturn_count = 0
    
    for idx in large_transactions.index:
        # 檢查交易後 24 小時內是否歸零
        next_events = events_df[events_df['time'] > events_df.loc[idx, 'time']]
        next_events = next_events[next_events['time'] <= events_df.loc[idx, 'time'] + timedelta(hours=24)]
        
        if len(next_events) > 0 and next_events['total_balance_twd'].min() < zero_threshold:
            uturn_count += 1
    
    uturn_ratio = uturn_count / len(events_df) if len(events_df) > 0 else 0
    
    # 6. 脈衝次數 (餘額從低到高再到低的次數)
    pulse_count = 0
    for i in range(1, len(events_df) - 1):
        prev_balance = events_df.iloc[i-1]['total_balance_twd']
        curr_balance = events_df.iloc[i]['total_balance_twd']
        next_balance = events_df.iloc[i+1]['total_balance_twd']
        
        # 檢測脈衝: 低 -> 高 -> 低
        if prev_balance < zero_threshold and curr_balance > zero_threshold * 10 and next_balance < zero_threshold:
            pulse_count += 1
    
    return {
        'zero_balance_count': zero_balance_count,
        'avg_balance_duration_hours': avg_balance_duration,
        'max_balance': max_balance,
        'balance_volatility': balance_volatility,
        'uturn_ratio': uturn_ratio,
        'pulse_count': pulse_count
    }

# 提取所有用戶的特徵
print("\n提取 U 型迴轉特徵...")
all_users = set(twd_transfer['user_id']) | set(usdt_trading['user_id']) | set(crypto_transfer['user_id'])
print(f"總共 {len(all_users)} 個用戶")

features_list = []
for user_id in list(all_users)[:1000]:  # 先處理前 1000 個用戶
    events_df = calculate_balance_timeline(user_id)
    features = extract_uturn_features(events_df)
    features['user_id'] = user_id
    features['label'] = user_labels.get(user_id, 0)
    features_list.append(features)

features_df = pd.DataFrame(features_list)

# 保存特徵
features_df.to_csv('visulize_result/outputs_uturn/uturn_features.csv', index=False)
print(f"\n特徵已保存: {len(features_df)} 個用戶")

# 統計分析
print("\n=== 特徵統計 ===")
print("\n正常用戶 (label=0):")
print(features_df[features_df['label'] == 0].describe())
print("\n黑名單用戶 (label=1):")
print(features_df[features_df['label'] == 1].describe())

# 視覺化 1: 特徵分佈對比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('U 型迴轉特徵分佈對比', fontsize=16, fontweight='bold')

features_to_plot = [
    ('zero_balance_count', '餘額歸零次數'),
    ('avg_balance_duration_hours', '平均餘額停留時間 (小時)'),
    ('max_balance', '最大餘額 (TWD)'),
    ('balance_volatility', '餘額波動率'),
    ('uturn_ratio', 'U 型迴轉比率'),
    ('pulse_count', '脈衝次數')
]

for idx, (feature, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    normal_data = features_df[features_df['label'] == 0][feature]
    blacklist_data = features_df[features_df['label'] == 1][feature]
    
    ax.hist(normal_data, bins=30, alpha=0.6, label='正常用戶', color='blue', edgecolor='black')
    ax.hist(blacklist_data, bins=30, alpha=0.6, label='黑名單用戶', color='red', edgecolor='black')
    
    ax.set_xlabel(title)
    ax.set_ylabel('用戶數量')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_distribution.png', dpi=300, bbox_inches='tight')
print("\n特徵分佈圖已保存")

# 視覺化 2: 箱型圖對比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('U 型迴轉特徵箱型圖對比', fontsize=16, fontweight='bold')

for idx, (feature, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    data_to_plot = [
        features_df[features_df['label'] == 0][feature],
        features_df[features_df['label'] == 1][feature]
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['正常用戶', '黑名單用戶'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visulize_result/outputs_uturn/feature_boxplot.png', dpi=300, bbox_inches='tight')
print("箱型圖已保存")

print("\n完成!")
