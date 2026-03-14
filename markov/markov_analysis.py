import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('markov/outputs', exist_ok=True)

# 讀取數據
print("讀取數據...")
user_info = pd.read_csv('data/user_info.csv')
trading = pd.read_csv('data/usdt_twd_trading.csv')
train_label = pd.read_csv('data/train_label.csv')

# 數據前處理
print("數據前處理...")
trading['source'] = trading['source'].astype(int)
user_info['user_source'] = user_info['user_source'].astype(int)

# 定義介面名稱映射
source_map = {0: 'WEB', 1: 'APP', 2: 'API'}

# 計算每個用戶的介面使用統計
print("計算介面使用統計...")
user_stats = []

for user_id in trading['user_id'].unique():
    user_trades = trading[trading['user_id'] == user_id]
    user_data = user_info[user_info['user_id'] == user_id]
    
    if len(user_data) == 0:
        continue
    
    # 註冊來源
    reg_source = user_data['user_source'].iloc[0]
    
    # 交易來源統計
    source_counts = user_trades['source'].value_counts()
    total_trades = len(user_trades)
    
    # 計算各介面使用比例
    web_ratio = source_counts.get(0, 0) / total_trades
    app_ratio = source_counts.get(1, 0) / total_trades
    api_ratio = source_counts.get(2, 0) / total_trades
    
    # 計算介面多樣性
    interface_diversity = (source_counts > 0).sum()
    
    # 計算狀態轉移次數
    sources = user_trades.sort_values('updated_at')['source'].values
    transitions = 0
    if len(sources) > 1:
        transitions = np.sum(sources[:-1] != sources[1:])
    
    # 轉移率
    transition_rate = transitions / total_trades if total_trades > 1 else 0
    
    # 獲取標籤 (1=黑名單, 0=正常)
    label_data = train_label[train_label['user_id'] == user_id]
    if len(label_data) > 0:
        label = label_data['status'].iloc[0]
        label_name = '黑名單' if label == 1 else '正常'
    else:
        label = -1
        label_name = '未標記'
    
    user_stats.append({
        'user_id': user_id,
        'label': label,
        'label_name': label_name,
        'reg_source': source_map[reg_source],
        'total_trades': total_trades,
        'web_ratio': web_ratio,
        'app_ratio': app_ratio,
        'api_ratio': api_ratio,
        'interface_diversity': interface_diversity,
        'transitions': transitions,
        'transition_rate': transition_rate,
        'dominant_source': source_map[source_counts.idxmax()],
        'api_only': 1 if api_ratio == 1.0 else 0
    })

df_stats = pd.DataFrame(user_stats)

# 保存統計結果
df_stats.to_csv('markov/outputs/user_interface_stats.csv', index=False)
print(f"\n統計結果已保存: {len(df_stats)} 個用戶")
print(f"  - 黑名單: {(df_stats['label'] == 1).sum()}")
print(f"  - 正常: {(df_stats['label'] == 0).sum()}")
print(f"  - 未標記: {(df_stats['label'] == -1).sum()}")
print(f"\nAPI 單一使用者: {df_stats['api_only'].sum()}")

# 只保留有標籤的用戶進行視覺化
df_labeled = df_stats[df_stats['label'] != -1].copy()
print(f"\n用於視覺化的標記用戶: {len(df_labeled)}")


# ============ 視覺化 ============
print("\n開始視覺化...")

# 定義顏色
colors = df_labeled['label'].map({0: 'blue', 1: 'red'})
labels_legend = df_labeled['label'].map({0: '正常', 1: '黑名單'})

# 1. 散佈圖：API 使用比例 vs 介面多樣性
fig, ax = plt.subplots(figsize=(12, 7))
for label_val, color, name in [(0, 'blue', '正常'), (1, 'red', '黑名單')]:
    mask = df_labeled['label'] == label_val
    ax.scatter(df_labeled[mask]['api_ratio'], 
               df_labeled[mask]['interface_diversity'],
               c=color,
               s=60,
               alpha=0.6,
               label=name,
               edgecolors='black',
               linewidth=0.5)

ax.set_xlabel('API 使用比例', fontsize=13, fontweight='bold')
ax.set_ylabel('介面多樣性 (使用介面數量)', fontsize=13, fontweight='bold')
ax.set_title('Markov 狀態轉移分析：API 使用 vs 介面多樣性', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['單一介面', '兩種介面', '三種介面'])
plt.tight_layout()
plt.savefig('markov/outputs/api_diversity_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 已生成: api_diversity_scatter.png")

# 2. 散佈圖：轉移率 vs API 使用比例
fig, ax = plt.subplots(figsize=(12, 7))
for label_val, color, name in [(0, 'blue', '正常'), (1, 'red', '黑名單')]:
    mask = df_labeled['label'] == label_val
    ax.scatter(df_labeled[mask]['api_ratio'], 
               df_labeled[mask]['transition_rate'],
               c=color,
               s=60,
               alpha=0.6,
               label=name,
               edgecolors='black',
               linewidth=0.5)

ax.set_xlabel('API 使用比例', fontsize=13, fontweight='bold')
ax.set_ylabel('介面轉移率', fontsize=13, fontweight='bold')
ax.set_title('Markov 狀態轉移分析：介面切換行為', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('markov/outputs/transition_rate_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 已生成: transition_rate_scatter.png")

# 3. 散佈圖：交易次數 vs API 使用比例
fig, ax = plt.subplots(figsize=(12, 7))
for label_val, color, name in [(0, 'blue', '正常'), (1, 'red', '黑名單')]:
    mask = df_labeled['label'] == label_val
    ax.scatter(df_labeled[mask]['total_trades'], 
               df_labeled[mask]['api_ratio'],
               c=color,
               s=60,
               alpha=0.6,
               label=name,
               edgecolors='black',
               linewidth=0.5)

ax.set_xlabel('交易次數 (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('API 使用比例', fontsize=13, fontweight='bold')
ax.set_title('Markov 狀態轉移分析：交易活躍度 vs API 依賴度', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('markov/outputs/trades_vs_api_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 已生成: trades_vs_api_ratio.png")

# 4. 散佈圖：WEB vs APP 使用比例
fig, ax = plt.subplots(figsize=(12, 7))
for label_val, color, name in [(0, 'blue', '正常'), (1, 'red', '黑名單')]:
    mask = df_labeled['label'] == label_val
    ax.scatter(df_labeled[mask]['web_ratio'], 
               df_labeled[mask]['app_ratio'],
               c=color,
               s=60,
               alpha=0.6,
               label=name,
               edgecolors='black',
               linewidth=0.5)

ax.set_xlabel('WEB 使用比例', fontsize=13, fontweight='bold')
ax.set_ylabel('APP 使用比例', fontsize=13, fontweight='bold')
ax.set_title('Markov 狀態轉移分析：WEB vs APP 使用分布', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('markov/outputs/web_vs_app_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 已生成: web_vs_app_scatter.png")

# 5. 散佈圖：介面轉移次數 vs 交易次數
fig, ax = plt.subplots(figsize=(12, 7))
for label_val, color, name in [(0, 'blue', '正常'), (1, 'red', '黑名單')]:
    mask = df_labeled['label'] == label_val
    data = df_labeled[mask]
    # 過濾掉 transitions = 0 的點以便使用 log scale
    data_filtered = data[data['transitions'] > 0]
    ax.scatter(data_filtered['total_trades'], 
               data_filtered['transitions'],
               c=color,
               s=60,
               alpha=0.6,
               label=name,
               edgecolors='black',
               linewidth=0.5)

ax.set_xlabel('交易次數 (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('介面轉移次數 (log scale)', fontsize=13, fontweight='bold')
ax.set_title('Markov 狀態轉移分析：交易量 vs 介面切換次數', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('markov/outputs/trades_vs_transitions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 已生成: trades_vs_transitions.png")

print("\n所有圖表已生成完成！")
print(f"輸出目錄: markov/outputs/")

# 輸出關鍵統計
print("\n=== 關鍵發現 ===")
blacklist = df_labeled[df_labeled['label'] == 1]
normal = df_labeled[df_labeled['label'] == 0]

print(f"\n黑名單用戶 API 使用比例: 平均 {blacklist['api_ratio'].mean():.2%}")
print(f"正常用戶 API 使用比例: 平均 {normal['api_ratio'].mean():.2%}")

print(f"\n黑名單用戶介面多樣性: 平均 {blacklist['interface_diversity'].mean():.2f}")
print(f"正常用戶介面多樣性: 平均 {normal['interface_diversity'].mean():.2f}")

print(f"\n黑名單用戶轉移率: 平均 {blacklist['transition_rate'].mean():.4f}")
print(f"正常用戶轉移率: 平均 {normal['transition_rate'].mean():.4f}")

print(f"\n黑名單中 API 單一使用者: {blacklist['api_only'].sum()} / {len(blacklist)} ({blacklist['api_only'].sum()/len(blacklist)*100:.1f}%)")
print(f"正常用戶中 API 單一使用者: {normal['api_only'].sum()} / {len(normal)} ({normal['api_only'].sum()/len(normal)*100:.1f}%)")
