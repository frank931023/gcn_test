"""
視覺化：User 幣種多樣性分析
探索「使用多種幣種」是否可作為分類特徵
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 載入資料 ──────────────────────────────────────────────
crypto = pd.read_csv('data/crypto_transfer.csv')
train_label = pd.read_csv('data/train_label.csv')

# status: 1 = 可疑, 0 = 正常 (假設)
labeled = train_label.copy()
labeled.columns = ['user_id', 'status']

# ── 計算每個 user 使用的幣種數量與種類 ────────────────────
user_currency = crypto.groupby('user_id')['currency'].agg(
    currency_count='nunique',
    currencies=lambda x: list(x.unique()),
    tx_count='count'
).reset_index()

# 合併標籤
df = user_currency.merge(labeled, on='user_id', how='left')
df['status'] = df['status'].fillna(-1).astype(int)  # -1 = unlabeled

labeled_df = df[df['status'] >= 0].copy()
labeled_df['label'] = labeled_df['status'].map({1: 'Suspicious', 0: 'Normal'})

print(f"Total users in crypto_transfer: {len(df)}")
print(f"Labeled users: {len(labeled_df)}")
print(f"  Suspicious: {(labeled_df['status']==1).sum()}")
print(f"  Normal:     {(labeled_df['status']==0).sum()}")
print(f"\nCurrency types found: {sorted(crypto['currency'].unique())}")
print(f"\nCurrency count distribution:")
print(df['currency_count'].value_counts().sort_index())

# ── 圖表 ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Currency Diversity Analysis per User', fontsize=15, fontweight='bold')

colors = {'Suspicious': '#e74c3c', 'Normal': '#3498db'}

# 1. 幣種數量分布（全體）
ax1 = axes[0, 0]
counts = df['currency_count'].value_counts().sort_index()
ax1.bar(counts.index, counts.values, color='#95a5a6', edgecolor='white')
ax1.set_title('Distribution of Currency Count (All Users)')
ax1.set_xlabel('Number of Currencies Used')
ax1.set_ylabel('User Count')
for i, (x, y) in enumerate(zip(counts.index, counts.values)):
    ax1.text(x, y + 0.5, str(y), ha='center', fontsize=8)

# 2. 幣種數量 by 標籤（有標籤的 user）
ax2 = axes[0, 1]
if len(labeled_df) > 0:
    for label, grp in labeled_df.groupby('label'):
        cnt = grp['currency_count'].value_counts().sort_index()
        ax2.plot(cnt.index, cnt.values, marker='o', label=label,
                 color=colors[label], linewidth=2)
    ax2.set_title('Currency Count Distribution by Label')
    ax2.set_xlabel('Number of Currencies Used')
    ax2.set_ylabel('User Count')
    ax2.legend()
else:
    ax2.text(0.5, 0.5, 'No labeled data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Currency Count Distribution by Label')

# 3. Boxplot：幣種數量 vs 標籤
ax3 = axes[1, 0]
if len(labeled_df) > 0:
    groups = [grp['currency_count'].values for _, grp in labeled_df.groupby('label')]
    group_labels = [lbl for lbl, _ in labeled_df.groupby('label')]
    bp = ax3.boxplot(groups, labels=group_labels, patch_artist=True)
    for patch, lbl in zip(bp['boxes'], group_labels):
        patch.set_facecolor(colors[lbl])
        patch.set_alpha(0.7)
    ax3.set_title('Currency Count Boxplot by Label')
    ax3.set_ylabel('Number of Currencies Used')

    # 印出統計
    for lbl, grp in labeled_df.groupby('label'):
        print(f"\n{lbl} - currency_count stats:")
        print(grp['currency_count'].describe())
else:
    ax3.text(0.5, 0.5, 'No labeled data', ha='center', va='center', transform=ax3.transAxes)

# 4. 各幣種使用頻率（哪些幣種最常被用）
ax4 = axes[1, 1]
currency_freq = crypto['currency'].value_counts()
bars = ax4.barh(currency_freq.index[::-1], currency_freq.values[::-1], color='#9b59b6', edgecolor='white')
ax4.set_title('Overall Currency Usage Frequency')
ax4.set_xlabel('Transaction Count')
for bar, val in zip(bars, currency_freq.values[::-1]):
    ax4.text(val + 10, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=8)

plt.tight_layout()
plt.savefig('mix/outputs/currency_diversity.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: mix/outputs/currency_diversity.png")

# ── 額外：Suspicious user 偏好哪些幣種？ ─────────────────
if len(labeled_df) > 0 and (labeled_df['status'] == 1).sum() > 0:
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle('Currency Preference: Suspicious vs Normal', fontsize=13, fontweight='bold')

    for ax, (lbl, grp) in zip(axes2, labeled_df.groupby('label')):
        user_ids = grp['user_id'].tolist()
        txs = crypto[crypto['user_id'].isin(user_ids)]
        freq = txs['currency'].value_counts()
        ax.bar(freq.index, freq.values, color=colors[lbl], edgecolor='white', alpha=0.85)
        ax.set_title(f'{lbl} Users — Currency Usage')
        ax.set_xlabel('Currency')
        ax.set_ylabel('Transaction Count')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('mix/outputs/currency_preference_by_label.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/currency_preference_by_label.png")
