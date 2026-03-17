"""
GraphSAGE 視覺化驗證腳本
驗證特徵在圖拓樸結構上的同質性 (Homophily)

三種視覺化方法：
1. 鄰居聚合效應散佈圖 (自身特徵 vs 鄰居平均特徵)
2. t-SNE 降維對比 (原始特徵 vs 聚合後特徵)
3. 核心子圖視覺化 (Ego-graph with PyVis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import os
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'graph_sage/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 載入資料
# ─────────────────────────────────────────────
print("=" * 60)
print("載入資料...")
print("=" * 60)

train_label = pd.read_csv('data/train_label.csv')
shared_ip_edges = pd.read_csv('data/edge/shared_ip_edges.csv')
uturn_features = pd.read_csv('uturn_classifier/features.csv')
features_df = pd.read_csv('data/features.csv')

# 黑名單集合
bad_users = set(train_label[train_label['status'] == 1]['user_id'].astype(str))
print(f"黑名單用戶數: {len(bad_users)}")
print(f"共用 IP 邊數: {len(shared_ip_edges)}")
print(f"U-turn 特徵用戶數: {len(uturn_features)}")

# 合併標籤到 uturn_features
uturn_features['user_id'] = uturn_features['user_id'].astype(str)
uturn_features['label'] = uturn_features['user_id'].isin(bad_users).astype(int)

# 合併 features_df 的 KYC 時間差特徵
features_df['user_id'] = features_df['user_id'].astype(str)
features_df['confirmed_at'] = pd.to_datetime(features_df['confirmed_at'], errors='coerce')
features_df['level2_finished_at'] = pd.to_datetime(features_df['level2_finished_at'], errors='coerce')
features_df['kyc_time_gap_hours'] = (
    features_df['level2_finished_at'] - features_df['confirmed_at']
).dt.total_seconds() / 3600
features_df['kyc_time_gap_hours'] = features_df['kyc_time_gap_hours'].fillna(0).clip(lower=0)

# 合併到 uturn_features
uturn_features = uturn_features.merge(
    features_df[['user_id', 'kyc_time_gap_hours', 'crypto_ip_count', 'usdt_ip_count']],
    on='user_id', how='left'
).fillna(0)

print(f"\n合併後特徵欄位: {list(uturn_features.columns)}")
print(f"黑名單: {uturn_features['label'].sum()}, 正常: {(uturn_features['label'] == 0).sum()}")

# ─────────────────────────────────────────────
# 2. 建立共用 IP 圖
# ─────────────────────────────────────────────
print("\n建立共用 IP 圖...")

G = nx.Graph()
shared_ip_edges['user_id_1'] = shared_ip_edges['user_id_1'].astype(str)
shared_ip_edges['user_id_2'] = shared_ip_edges['user_id_2'].astype(str)

for _, row in shared_ip_edges.iterrows():
    G.add_edge(row['user_id_1'], row['user_id_2'], ip=row['source_ip_hash'])

print(f"圖節點數: {G.number_of_nodes()}, 邊數: {G.number_of_edges()}")

# 建立用戶特徵字典
user_features = uturn_features.set_index('user_id')[
    ['uturn_ratio', 'pulse_count', 'kyc_time_gap_hours']
].to_dict('index')

# ─────────────────────────────────────────────
# 3. 計算鄰居平均特徵 (模擬 GraphSAGE 第一層)
# ─────────────────────────────────────────────
print("\n計算鄰居聚合特徵...")

records = []
for user_id, feats in user_features.items():
    if user_id not in G:
        continue
    neighbors = list(G.neighbors(user_id))
    if len(neighbors) == 0:
        continue

    neighbor_uturn = []
    neighbor_pulse = []
    neighbor_kyc = []
    for nb in neighbors:
        if nb in user_features:
            neighbor_uturn.append(user_features[nb]['uturn_ratio'])
            neighbor_pulse.append(user_features[nb]['pulse_count'])
            neighbor_kyc.append(user_features[nb]['kyc_time_gap_hours'])

    if len(neighbor_uturn) == 0:
        continue

    records.append({
        'user_id': user_id,
        'self_uturn': feats['uturn_ratio'],
        'self_pulse': feats['pulse_count'],
        'self_kyc': feats['kyc_time_gap_hours'],
        'neighbor_uturn_mean': np.mean(neighbor_uturn),
        'neighbor_pulse_mean': np.mean(neighbor_pulse),
        'neighbor_kyc_mean': np.mean(neighbor_kyc),
        'neighbor_count': len(neighbors),
        'label': 1 if user_id in bad_users else 0,
    })

agg_df = pd.DataFrame(records)
print(f"有鄰居的用戶數: {len(agg_df)}")
print(f"  黑名單: {agg_df['label'].sum()}, 正常: {(agg_df['label'] == 0).sum()}")

# ─────────────────────────────────────────────
# 方法 1：鄰居聚合效應散佈圖
# ─────────────────────────────────────────────
print("\n[方法 1] 繪製鄰居聚合效應散佈圖...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('GraphSAGE 鄰居聚合效應驗證\n(自身特徵 vs 一階鄰居平均特徵)', fontsize=15, fontweight='bold')

colors = {0: '#4C72B0', 1: '#DD4444'}
labels_map = {0: '正常用戶', 1: '黑名單用戶'}
sizes = {0: 30, 1: 80}
alphas = {0: 0.4, 1: 0.85}

for label_val in [0, 1]:
    subset = agg_df[agg_df['label'] == label_val]
    axes[0].scatter(
        subset['self_uturn'], subset['neighbor_uturn_mean'],
        c=colors[label_val], s=sizes[label_val], alpha=alphas[label_val],
        label=f"{labels_map[label_val]} (n={len(subset)})",
        edgecolors='white' if label_val == 0 else 'black', linewidths=0.5
    )

axes[0].set_xlabel('自身 U-turn 比率', fontsize=12)
axes[0].set_ylabel('鄰居平均 U-turn 比率', fontsize=12)
axes[0].set_title('U-turn 比率：自身 vs 鄰居', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
# 標示右上角區域
axes[0].axhline(y=agg_df['neighbor_uturn_mean'].quantile(0.75), color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=agg_df['self_uturn'].quantile(0.75), color='gray', linestyle='--', alpha=0.5)
axes[0].text(0.98, 0.98, '高風險區域', transform=axes[0].transAxes,
             ha='right', va='top', fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

for label_val in [0, 1]:
    subset = agg_df[agg_df['label'] == label_val]
    axes[1].scatter(
        subset['self_pulse'], subset['neighbor_pulse_mean'],
        c=colors[label_val], s=sizes[label_val], alpha=alphas[label_val],
        label=f"{labels_map[label_val]} (n={len(subset)})",
        edgecolors='white' if label_val == 0 else 'black', linewidths=0.5
    )

axes[1].set_xlabel('自身脈衝次數', fontsize=12)
axes[1].set_ylabel('鄰居平均脈衝次數', fontsize=12)
axes[1].set_title('脈衝次數：自身 vs 鄰居', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=agg_df['neighbor_pulse_mean'].quantile(0.75), color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=agg_df['self_pulse'].quantile(0.75), color='gray', linestyle='--', alpha=0.5)
axes[1].text(0.98, 0.98, '高風險區域', transform=axes[1].transAxes,
             ha='right', va='top', fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out_path = f'{OUTPUT_DIR}/01_neighbor_aggregation_scatter.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已儲存: {out_path}")

# ─────────────────────────────────────────────
# 方法 2：t-SNE 降維對比 (原始 vs 聚合後)
# ─────────────────────────────────────────────
print("\n[方法 2] 繪製 t-SNE 降維對比圖...")

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # 準備特徵矩陣 (只用有鄰居的用戶)
    feature_cols = ['self_uturn', 'self_pulse', 'self_kyc']
    agg_cols = ['self_uturn', 'self_pulse', 'self_kyc',
                'neighbor_uturn_mean', 'neighbor_pulse_mean', 'neighbor_kyc_mean']

    X_raw = agg_df[feature_cols].values
    X_agg = agg_df[agg_cols].values
    y_labels = agg_df['label'].values

    scaler = StandardScaler()
    X_raw_scaled = scaler.fit_transform(X_raw)
    X_agg_scaled = scaler.fit_transform(X_agg)

    # 取樣加速 t-SNE (最多 500 筆)
    MAX_TSNE = 500
    if len(agg_df) > MAX_TSNE:
        bad_idx = np.where(y_labels == 1)[0]
        normal_idx = np.where(y_labels == 0)[0]
        n_normal = min(MAX_TSNE - len(bad_idx), len(normal_idx))
        normal_sample = np.random.RandomState(42).choice(normal_idx, n_normal, replace=False)
        sample_idx = np.concatenate([bad_idx, normal_sample])
        X_raw_scaled = X_raw_scaled[sample_idx]
        X_agg_scaled = X_agg_scaled[sample_idx]
        y_labels = y_labels[sample_idx]

    perplexity = min(30, len(y_labels) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=500)
    X_raw_2d = tsne.fit_transform(X_raw_scaled)

    tsne2 = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=500)
    X_agg_2d = tsne2.fit_transform(X_agg_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('t-SNE 降維對比：原始特徵 vs 加入圖結構聚合後', fontsize=15, fontweight='bold')

    for ax, X_2d, title in [
        (axes[0], X_raw_2d, '圖 A：原始特徵 (LightGBM 視角)'),
        (axes[1], X_agg_2d, '圖 B：加入鄰居聚合後 (GNN 視角)')
    ]:
        for label_val in [0, 1]:
            mask = y_labels == label_val
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors[label_val], s=sizes[label_val], alpha=alphas[label_val],
                label=f"{labels_map[label_val]} (n={mask.sum()})",
                edgecolors='white' if label_val == 0 else 'black', linewidths=0.5
            )
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.set_xlabel('t-SNE 維度 1', fontsize=11)
        ax.set_ylabel('t-SNE 維度 2', fontsize=11)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/02_tsne_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存: {out_path}")

except ImportError:
    print("  sklearn 未安裝，跳過 t-SNE 圖")

# ─────────────────────────────────────────────
# 方法 2b：KYC 時間差同質性熱圖
# ─────────────────────────────────────────────
print("\n[方法 2b] 繪製 KYC 時間差同質性熱圖...")

# 找出共用同一 IP 的群組，分析 KYC 時間差的相似性
ip_groups = shared_ip_edges.groupby('source_ip_hash').apply(
    lambda x: list(set(x['user_id_1'].tolist() + x['user_id_2'].tolist()))
).reset_index()
ip_groups.columns = ['ip_hash', 'users']
ip_groups['group_size'] = ip_groups['users'].apply(len)

# 只取群組大小 >= 3 的 IP
large_groups = ip_groups[ip_groups['group_size'] >= 3].head(20)

kyc_data = []
for _, row in large_groups.iterrows():
    users = row['users']
    kyc_vals = []
    labels_in_group = []
    for u in users:
        if u in user_features:
            kyc_vals.append(user_features[u].get('kyc_time_gap_hours', 0))
            labels_in_group.append(1 if u in bad_users else 0)
    if len(kyc_vals) >= 2:
        kyc_data.append({
            'ip_hash': row['ip_hash'][:8] + '...',
            'group_size': row['group_size'],
            'kyc_std': np.std(kyc_vals),
            'kyc_mean': np.mean(kyc_vals),
            'bad_ratio': np.mean(labels_in_group),
            'has_bad': any(l == 1 for l in labels_in_group)
        })

if kyc_data:
    kyc_df = pd.DataFrame(kyc_data).sort_values('bad_ratio', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('共用 IP 群組的 KYC 時間差分析\n(詐騙集團通常批次處理 KYC)', fontsize=14, fontweight='bold')

    # 左圖：KYC 標準差 (越低代表越同質)
    bar_colors = ['#DD4444' if r > 0 else '#4C72B0' for r in kyc_df['bad_ratio']]
    axes[0].barh(range(len(kyc_df)), kyc_df['kyc_std'], color=bar_colors, edgecolor='black', alpha=0.8)
    axes[0].set_yticks(range(len(kyc_df)))
    axes[0].set_yticklabels([f"IP:{h} (n={s})" for h, s in zip(kyc_df['ip_hash'], kyc_df['group_size'])], fontsize=9)
    axes[0].set_xlabel('KYC 時間差標準差 (小時)', fontsize=11)
    axes[0].set_title('KYC 時間差標準差\n(紅=含黑名單, 藍=全正常)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')

    # 右圖：黑名單比例 vs 群組大小
    scatter_colors = ['#DD4444' if r > 0 else '#4C72B0' for r in kyc_df['bad_ratio']]
    axes[1].scatter(kyc_df['group_size'], kyc_df['bad_ratio'],
                    c=scatter_colors, s=100, alpha=0.8, edgecolors='black')
    axes[1].set_xlabel('共用 IP 群組大小', fontsize=11)
    axes[1].set_ylabel('黑名單用戶比例', fontsize=11)
    axes[1].set_title('群組大小 vs 黑名單比例', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    red_patch = mpatches.Patch(color='#DD4444', label='含黑名單群組')
    blue_patch = mpatches.Patch(color='#4C72B0', label='全正常群組')
    axes[1].legend(handles=[red_patch, blue_patch], fontsize=11)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/03_kyc_time_gap_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存: {out_path}")

# ─────────────────────────────────────────────
# 方法 3：核心子圖視覺化 (Ego-graph)
# ─────────────────────────────────────────────
print("\n[方法 3] 繪製核心子圖視覺化...")

# 找出高風險節點 (脈衝次數最高 or U-turn 最高)
high_risk_users = uturn_features[
    (uturn_features['pulse_count'] > 0) | (uturn_features['uturn_ratio'] > 0.3)
].sort_values(['pulse_count', 'uturn_ratio'], ascending=False).head(50)

high_risk_set = set(high_risk_users['user_id'].astype(str).tolist())

# 建立子圖：高風險節點 + 其一階鄰居
subgraph_nodes = set()
for node in high_risk_set:
    if node in G:
        subgraph_nodes.add(node)
        subgraph_nodes.update(G.neighbors(node))

subgraph_nodes = list(subgraph_nodes)[:300]  # 限制節點數避免過密
H = G.subgraph(subgraph_nodes).copy()

print(f"  子圖節點數: {H.number_of_nodes()}, 邊數: {H.number_of_edges()}")

if H.number_of_nodes() > 0:
    # 計算節點屬性
    node_colors = []
    node_sizes = []
    node_labels_dict = {}

    for node in H.nodes():
        is_bad = node in bad_users
        is_high_risk = node in high_risk_set

        if is_bad:
            node_colors.append('#DD2222')
        elif is_high_risk:
            node_colors.append('#FF8800')
        else:
            node_colors.append('#4C72B0')

        # 節點大小依脈衝次數
        if node in user_features:
            pulse = user_features[node].get('pulse_count', 0)
            node_sizes.append(max(50, min(800, 50 + pulse * 200)))
        else:
            node_sizes.append(50)

    # 使用 spring layout
    pos = nx.spring_layout(H, seed=42, k=0.8)

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_facecolor('#F8F8F8')
    fig.patch.set_facecolor('#F8F8F8')

    # 畫邊
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, edge_color='#888888', width=0.8)

    # 畫節點
    nx.draw_networkx_nodes(H, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85)

    # 只標示黑名單節點的 ID
    bad_labels = {n: n[:6] for n in H.nodes() if n in bad_users}
    nx.draw_networkx_labels(H, pos, labels=bad_labels, ax=ax,
                            font_size=7, font_color='white', font_weight='bold')

    # 圖例
    legend_elements = [
        mpatches.Patch(color='#DD2222', label='黑名單用戶'),
        mpatches.Patch(color='#FF8800', label='高風險用戶 (高脈衝/U-turn)'),
        mpatches.Patch(color='#4C72B0', label='正常用戶'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
              framealpha=0.9, fancybox=True)

    ax.set_title(
        f'核心子圖視覺化 (Ego-graph)\n'
        f'節點大小 = 脈衝次數 | 節點數: {H.number_of_nodes()} | 邊數: {H.number_of_edges()}',
        fontsize=14, fontweight='bold'
    )
    ax.axis('off')

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/04_ego_subgraph.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存: {out_path}")

# ─────────────────────────────────────────────
# 方法 3b：PyVis 互動式子圖 (HTML)
# ─────────────────────────────────────────────
print("\n[方法 3b] 嘗試生成 PyVis 互動式子圖...")

try:
    from pyvis.network import Network

    # 取最核心的子圖 (限制 100 節點)
    core_nodes = list(subgraph_nodes)[:100]
    H_small = G.subgraph(core_nodes).copy()

    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.barnes_hut()

    for node in H_small.nodes():
        is_bad = node in bad_users
        is_high_risk = node in high_risk_set

        if is_bad:
            color = '#FF4444'
            group = 'blacklist'
        elif is_high_risk:
            color = '#FF8800'
            group = 'high_risk'
        else:
            color = '#4488FF'
            group = 'normal'

        pulse = user_features.get(node, {}).get('pulse_count', 0)
        uturn = user_features.get(node, {}).get('uturn_ratio', 0)
        size = max(10, min(50, 10 + pulse * 15))

        net.add_node(
            node,
            label=node[:8] if is_bad else '',
            color=color,
            size=size,
            title=f"用戶: {node}<br>黑名單: {'是' if is_bad else '否'}<br>脈衝次數: {pulse}<br>U-turn: {uturn:.3f}",
            group=group
        )

    for u, v in H_small.edges():
        net.add_edge(u, v, color='#555555', width=1)

    html_path = f'{OUTPUT_DIR}/05_interactive_subgraph.html'
    net.save_graph(html_path)
    print(f"  已儲存互動式圖: {html_path}")

except ImportError:
    print("  pyvis 未安裝，跳過互動式圖 (可執行: pip install pyvis)")

# ─────────────────────────────────────────────
# 方法 4：同質性指數 (Homophily Score) 統計圖
# ─────────────────────────────────────────────
print("\n[方法 4] 計算並視覺化同質性指數...")

# 計算每條邊的同質性 (兩端節點是否同為黑名單)
edge_types = {'bad-bad': 0, 'bad-normal': 0, 'normal-normal': 0}
for u, v in G.edges():
    u_bad = u in bad_users
    v_bad = v in bad_users
    if u_bad and v_bad:
        edge_types['bad-bad'] += 1
    elif u_bad or v_bad:
        edge_types['bad-normal'] += 1
    else:
        edge_types['normal-normal'] += 1

total_edges = sum(edge_types.values())
bad_node_ratio = len(bad_users) / max(G.number_of_nodes(), 1)
expected_bad_bad = bad_node_ratio ** 2
actual_bad_bad = edge_types['bad-bad'] / max(total_edges, 1)
homophily_score = actual_bad_bad / max(expected_bad_bad, 1e-9)

print(f"\n  邊類型分布:")
for k, v in edge_types.items():
    print(f"    {k}: {v} ({v/max(total_edges,1)*100:.1f}%)")
print(f"  黑名單節點比例: {bad_node_ratio:.4f}")
print(f"  期望 bad-bad 邊比例 (隨機): {expected_bad_bad:.4f}")
print(f"  實際 bad-bad 邊比例: {actual_bad_bad:.4f}")
print(f"  同質性倍數 (Homophily Ratio): {homophily_score:.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('圖結構同質性分析 (Homophily Analysis)', fontsize=14, fontweight='bold')

# 左圖：邊類型分布
edge_labels = ['黑名單-黑名單\n(bad-bad)', '黑名單-正常\n(bad-normal)', '正常-正常\n(normal-normal)']
edge_values = [edge_types['bad-bad'], edge_types['bad-normal'], edge_types['normal-normal']]
edge_colors = ['#DD2222', '#FF8800', '#4C72B0']
bars = axes[0].bar(edge_labels, edge_values, color=edge_colors, edgecolor='black', alpha=0.85)
axes[0].set_ylabel('邊的數量', fontsize=12)
axes[0].set_title('共用 IP 邊的類型分布', fontsize=13)
for bar, val in zip(bars, edge_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val}\n({val/max(total_edges,1)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# 右圖：實際 vs 期望 bad-bad 邊比例
categories = ['期望 (隨機圖)', '實際觀測']
values = [expected_bad_bad * 100, actual_bad_bad * 100]
bar_colors_2 = ['#888888', '#DD2222']
bars2 = axes[1].bar(categories, values, color=bar_colors_2, edgecolor='black', alpha=0.85, width=0.5)
axes[1].set_ylabel('bad-bad 邊比例 (%)', fontsize=12)
axes[1].set_title(f'同質性驗證\n(實際/期望 = {homophily_score:.1f}x)', fontsize=13)
for bar, val in zip(bars2, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

annotation = f"同質性倍數: {homophily_score:.1f}x\n{'✓ 存在明顯同質性' if homophily_score > 2 else '△ 同質性不顯著'}"
axes[1].text(0.98, 0.98, annotation, transform=axes[1].transAxes,
             ha='right', va='top', fontsize=11,
             color='red' if homophily_score > 2 else 'gray',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
out_path = f'{OUTPUT_DIR}/06_homophily_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已儲存: {out_path}")

# ─────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("所有視覺化完成！")
print("=" * 60)
print(f"\n輸出目錄: {OUTPUT_DIR}/")
print("  01_neighbor_aggregation_scatter.png  - 鄰居聚合效應散佈圖")
print("  02_tsne_comparison.png               - t-SNE 降維對比")
print("  03_kyc_time_gap_analysis.png         - KYC 時間差同質性分析")
print("  04_ego_subgraph.png                  - 核心子圖視覺化")
print("  05_interactive_subgraph.html         - PyVis 互動式子圖 (需 pyvis)")
print("  06_homophily_analysis.png            - 同質性指數統計圖")
