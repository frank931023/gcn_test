import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
import copy
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import math

# v7: add dormant_intensity, is_isolated, is_round_amount; scale embeddings for isolated nodes

def build_graph_data():
    shared_ip = pd.read_csv('../data/edge/shared_ip_edges.csv')
    crypto = pd.read_csv('../data/crypto_transfer.csv')
    from collections import defaultdict
    edge_stats = defaultdict(lambda: {'count': 0, 'amount': 0.0})
    for _, row in shared_ip.iterrows():
        try:
            u = int(row['user_id_1']); v = int(row['user_id_2'])
        except Exception:
            continue
        k = (min(u, v), max(u, v))
        edge_stats[k]['count'] += 1
    crypto_internal = crypto[crypto['relation_user_id'].notnull()]
    for _, row in crypto_internal.iterrows():
        try:
            u = int(row['user_id']); v = int(row['relation_user_id'])
        except Exception:
            continue
        amt = 0.0
        if 'amount' in row.index:
            try:
                amt = float(row['amount'])
            except Exception:
                amt = 0.0
        k = (min(u, v), max(u, v))
        edge_stats[k]['count'] += 1
        edge_stats[k]['amount'] += amt
    G = nx.Graph()
    for (u, v), s in edge_stats.items():
        G.add_edge(u, v, count=s['count'], amount=s['amount'])
    all_nodes = sorted(list(G.nodes()))
    id_map = {n: i for i, n in enumerate(all_nodes)}
    edge_list = []
    for u, v in G.edges():
        edge_list.append([id_map[u], id_map[v]])
        edge_list.append([id_map[v], id_map[u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return G, id_map, edge_index, all_nodes


def try_parse_datetime_series(s):
    return pd.to_datetime(s, errors='coerce', utc=True)


def find_first_tx_times(user_ids):
    tx_files = ['../data/crypto_transfer.csv','../data/twd_transfer.csv','../data/usdt_twd_trading.csv','../data/usdt_swap.csv']
    first_times = {u: pd.NaT for u in user_ids}
    for f in tx_files:
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        date_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'created' in c.lower()]
        if len(date_cols) == 0:
            continue
        user_cols = [c for c in df.columns if 'user' in c.lower() and ('id' in c.lower() or 'user' in c.lower())]
        if len(user_cols) == 0:
            continue
        for uc in user_cols:
            for dc in date_cols:
                try:
                    tmp = df[[uc, dc]].dropna()
                    tmp[dc] = try_parse_datetime_series(tmp[dc])
                    for uid, grp in tmp.groupby(uc):
                        if uid not in first_times:
                            continue
                        tmin = grp[dc].min()
                        if pd.isna(first_times[uid]) or (not pd.isna(tmin) and tmin < first_times[uid]):
                            first_times[uid] = tmin
                except Exception:
                    continue
    return first_times


def compute_round_amount_ratio(user_ids):
    # proportion of transactions with amount % 10000 == 0 per user
    tx_files = ['../data/crypto_transfer.csv','../data/twd_transfer.csv','../data/usdt_twd_trading.csv','../data/usdt_swap.csv']
    counts = {u: 0 for u in user_ids}
    round_counts = {u: 0 for u in user_ids}
    for f in tx_files:
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        amt_cols = [c for c in df.columns if 'amount' in c.lower()]
        user_cols = [c for c in df.columns if 'user' in c.lower() and ('id' in c.lower() or 'user' in c.lower())]
        if len(amt_cols) == 0 or len(user_cols) == 0:
            continue
        a = amt_cols[0]
        for uc in user_cols:
            tmp = df[[uc, a]].dropna()
            for uid, grp in tmp.groupby(uc):
                if uid not in counts:
                    continue
                vals = pd.to_numeric(grp[a], errors='coerce').dropna().astype(float)
                if len(vals) == 0:
                    continue
                counts[uid] += len(vals)
                round_counts[uid] += (np.isclose(vals % 10000, 0)).sum()
    ratios = {u: (round_counts[u] / counts[u]) if counts[u] > 0 else 0.0 for u in user_ids}
    return ratios


def prepare_features_df_v7(id_map, all_nodes, G):
    behavior_df = pd.read_csv('extended_features_analysis.csv')
    n2v_df = pd.read_csv('user_n2v.embeddings', sep=' ', skiprows=1, header=None)
    n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]
    final_df = behavior_df.merge(n2v_df, on='user_id', how='inner')

    # amount transform
    amount_cols = [c for c in final_df.columns if 'amount' in c.lower() or 'total' in c.lower()]
    for c in amount_cols:
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0.0)
        final_df[c] = np.log1p(final_df[c])

    # PCA on n2v
    n2v_cols = [c for c in final_df.columns if c.startswith('n2v_')]
    if len(n2v_cols) > 0:
        pca = PCA(n_components=16, random_state=42)
        n2v_vals = final_df[n2v_cols].fillna(0).values
        n2v_red = pca.fit_transform(n2v_vals)
        for i in range(n2v_red.shape[1]):
            final_df[f'n2v_pca_{i}'] = n2v_red[:, i]
        final_df = final_df.drop(columns=n2v_cols)

    final_df = final_df.set_index('user_id')

    # neighbor risk and neighbor n2v avg and degree
    status_dict = final_df['status'].to_dict() if 'status' in final_df.columns else {}
    neigh_risk = {}
    neigh_n2v_avg = {}
    degree_map = {}
    for node in all_nodes:
        degree_map[node] = G.degree(node) if node in G else 0
        if node not in final_df.index:
            neigh_risk[node] = 0.0
            neigh_n2v_avg[node] = np.zeros(16)
            continue
        neighbors = list(G.neighbors(node)) if node in G else []
        if len(neighbors) == 0:
            neigh_risk[node] = 0.0
            neigh_n2v_avg[node] = np.zeros(16)
            continue
        cnt_black = sum(1 for nb in neighbors if status_dict.get(nb, 0) == 1)
        neigh_risk[node] = cnt_black / max(1, len(neighbors))
        vecs = []
        for nb in neighbors:
            if nb in final_df.index:
                v = final_df.loc[nb, [c for c in final_df.columns if c.startswith('n2v_pca_')]].values
                vecs.append(v)
        if len(vecs) > 0:
            neigh_n2v_avg[node] = np.mean(np.stack(vecs), axis=0)
        else:
            neigh_n2v_avg[node] = np.zeros(16)

    final_df['neighbor_risk_score'] = pd.Series(neigh_risk)
    for i in range(16):
        final_df[f'nb_n2v_{i}'] = pd.Series({k: neigh_n2v_avg[k][i] for k in neigh_n2v_avg})

    # account tenure
    date_cols = [c for c in final_df.columns if 'level1_finished_at' in c.lower() or 'confirmed' in c.lower() or 'created_at' in c.lower()]
    if len(date_cols) > 0:
        for c in date_cols:
            try:
                final_df[c] = pd.to_datetime(final_df[c], errors='coerce', utc=True)
            except Exception:
                final_df[c] = pd.to_datetime(final_df[c], errors='coerce')
        ref_dates = pd.concat([final_df[c].dropna() for c in date_cols]) if len(date_cols) > 0 else pd.Series([])
        if len(ref_dates) > 0:
            ref = ref_dates.max()
            final_df['account_tenure_days'] = (ref - final_df[date_cols[0]]).dt.days.fillna(0).clip(lower=0)
        else:
            final_df['account_tenure_days'] = 0
    else:
        final_df['account_tenure_days'] = 0

    # first_tx and delay
    user_ids = list(final_df.index)
    first_tx = find_first_tx_times(user_ids)
    final_df['first_tx_time'] = pd.Series(first_tx)
    if 'level2_finished_at' in final_df.columns:
        try:
            final_df['level2_finished_at'] = pd.to_datetime(final_df['level2_finished_at'], errors='coerce', utc=True)
            final_df['first_tx_delay_days'] = (final_df['first_tx_time'] - final_df['level2_finished_at']).dt.days.fillna(9999).clip(lower=0)
        except Exception:
            final_df['first_tx_delay_days'] = 9999
    else:
        final_df['first_tx_delay_days'] = 9999

    # round amount ratio
    round_ratios = compute_round_amount_ratio(user_ids)
    final_df['is_round_amount_ratio'] = pd.Series(round_ratios)

    # dormant intensity = first_tx_delay_days * swap_twd_avg (use 0 if missing)
    if 'swap_twd_avg' in final_df.columns:
        final_df['dormant_intensity'] = final_df['first_tx_delay_days'] * final_df['swap_twd_avg']
    else:
        final_df['dormant_intensity'] = final_df['first_tx_delay_days'] * 0.0

    # is_isolated based on degree_map
    final_df['shared_ip_degree'] = pd.Series(degree_map)
    final_df['is_isolated'] = (final_df['shared_ip_degree'] < 2).astype(int)

    # coerce objects
    for col in list(final_df.columns):
        if final_df[col].dtype == object:
            dt = pd.to_datetime(final_df[col], errors='coerce', utc=True)
            if dt.notna().any():
                final_df[col] = (dt.astype('int64') // 10**9).fillna(0)
            else:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

    if 'status' in final_df.columns:
        final_df['status'] = pd.to_numeric(final_df['status'], errors='coerce').fillna(0).astype(int)

    feat_cols = [c for c in final_df.columns if c != 'status']
    def _to_numeric_val(v):
        try:
            if pd.isna(v):
                return 0.0
        except Exception:
            pass
        if isinstance(v, (pd.Timestamp,)) or isinstance(v, np.datetime64):
            try:
                return float(pd.Timestamp(v).timestamp())
            except Exception:
                return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0
    if len(feat_cols) > 0:
        final_df[feat_cols] = final_df[feat_cols].applymap(_to_numeric_val)

    X = np.zeros((len(all_nodes), len(feat_cols)), dtype=float)
    y = np.zeros(len(all_nodes), dtype=int)
    for i, node in enumerate(all_nodes):
        if node in final_df.index:
            row = final_df.loc[node]
            vals = row[feat_cols].values.astype(float)
            X[i, :] = vals
            y[i] = int(row.get('status', 0))
        else:
            X[i, :] = 0.0
            y[i] = 0

    feature_names = feat_cols
    X = pd.DataFrame(X, index=all_nodes, columns=feature_names)
    y = pd.Series(y, index=all_nodes)
    return final_df, X, y, feature_names


class EnhancedGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=1, dropout=0.3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.elu(h1)
        h2 = self.conv2(h1, edge_index)
        h2 = F.elu(h2)
        combined = torch.cat([h2, x], dim=1)
        return self.classifier(combined)


def train_with_earlystop(model, data, criterion, optimizer, patience=20, max_epochs=300):
    best_val = 1e9
    patience_cnt = 0
    best_state = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(data.x, data.edge_index)
            val_loss = criterion(v_logits[data.val_mask], data.y[data.val_mask]).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_cnt = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
        if patience_cnt >= patience:
            print('Early stopping at epoch', epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def find_best_threshold_by_f1(val_probs, y_val):
    best_f1 = -1.0
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        p = (val_probs >= t).astype(int)
        f = f1_score(y_val, p, zero_division=0)
        if f > best_f1:
            best_f1 = f; best_t = t
    return best_t, best_f1


def plot_feature_distribution(df, target_col='status', out_fname='feature_distribution_analysis.png'):
    # df: DataFrame with 'user_id' (or index) and target_col indicating 0/1 label
    exclude_cols = ['user_id', 'actual_status', 'pred_status', 'pred_prob', 'status', 'confirmed_at',
                    'level1_finished_at', 'level2_finished_at']
    # select numeric columns only
    num_cols = [c for c in df.columns if (pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])) and c not in exclude_cols]
    if len(num_cols) == 0:
        print('No numeric features found for distribution plot.')
        return

    num_features = len(num_cols)
    cols = 4
    rows = math.ceil(num_features / cols)
    plt.figure(figsize=(20, 5 * rows))
    sns.set_style('whitegrid')

    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols, i + 1)
        try:
            # density plots with independent normalization so class imbalance doesn't hide differences
            sns.kdeplot(data=df[df[target_col] == 0], x=col, label='Normal', fill=True, color='skyblue', common_norm=False)
            sns.kdeplot(data=df[df[target_col] == 1], x=col, label='Blacklist', fill=True, color='red', common_norm=False)
            plt.title(f'Dist: {col}', fontsize=12)
            plt.xlabel('')
            plt.ylabel('Density')
            # if range is large, use symmetric log scale for readability
            try:
                col_min = df[col].min()
                col_max = df[col].max()
                if pd.notna(col_max) and col_max > 1000 and (pd.notna(col_min) and col_min >= 0):
                    plt.xscale('symlog')
                    plt.title(f'Dist: {col} (Log Scale)', fontsize=12)
            except Exception:
                pass
            plt.legend()
        except Exception as e:
            plt.text(0.5, 0.5, f'Error plotting {col}: {e}', ha='center')

    plt.tight_layout()
    try:
        plt.savefig(out_fname, dpi=150)
        print(f'分析圖表已儲存為 {out_fname}')
    except Exception as e:
        print('Failed to save distribution plot:', e)


if __name__ == '__main__':
    print('Building graph and features (v7)...')
    G, id_map, edge_index, all_nodes = build_graph_data()
    final_df, X_df, y_series, feature_names = prepare_features_df_v7(id_map, all_nodes, G)

    idx = np.arange(len(X_df))
    y_np = y_series.values
    tr_idx, temp_idx = train_test_split(idx, stratify=y_np, test_size=0.3, random_state=42)
    val_idx, te_idx = train_test_split(temp_idx, stratify=y_np[temp_idx], test_size=0.5, random_state=42)

    train_mask = torch.zeros(len(X_df), dtype=torch.bool)
    val_mask = torch.zeros(len(X_df), dtype=torch.bool)
    test_mask = torch.zeros(len(X_df), dtype=torch.bool)
    train_mask[tr_idx] = True; val_mask[val_idx] = True; test_mask[te_idx] = True

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_series.values, dtype=torch.long)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = train_mask; data.val_mask = val_mask; data.test_mask = test_mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = EnhancedGAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

    print('Training GAT encoder (v7)...')
    model = train_with_earlystop(model, data, criterion, optimizer, patience=30, max_epochs=200)

    model.eval()
    with torch.no_grad():
        emb = F.elu(model.conv1(data.x, data.edge_index)).cpu().numpy()

    # behavioral fallback: scale down embeddings for isolated nodes
    is_isolated = X_df['is_isolated'].values if 'is_isolated' in X_df.columns else np.zeros(len(X_df))
    # scale factor: isolated -> 0.1, others -> 1.0
    scale_f = np.where(is_isolated == 1, 0.1, 1.0)
    emb_scaled = emb * scale_f[:, None]

    # stacking: embeddings (scaled) + raw features
    X_stack = np.hstack([emb_scaled, X_scaled])
    X_tr = X_stack[tr_idx]; y_tr = y_np[tr_idx]
    X_val = X_stack[val_idx]; y_val = y_np[val_idx]
    X_te = X_stack[te_idx]; y_te = y_np[te_idx]

    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=500, class_weight='balanced', random_state=42)
    except Exception:
        clf = HistGradientBoostingClassifier(max_iter=200)

    clf.fit(X_tr, y_tr)
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold_by_f1(val_probs, y_val)
    print(f'Best threshold on val by F1: {best_t:.4f}, val F1: {best_f1:.4f}')

    test_probs = clf.predict_proba(X_te)[:, 1]
    test_preds = (test_probs >= best_t).astype(int)
    cm = confusion_matrix(y_te, test_preds)
    print('\nStacked model Test Confusion Matrix:\n', cm)
    print('\nStacked Classification Report:')
    print(classification_report(y_te, test_preds))

    # save artifacts
    try:
        pd.Series(best_t, index=['best_threshold']).to_csv('gcn_v7_best_threshold.csv')
        torch.save(model.state_dict(), 'gcn_v7_gat.pt')
    except Exception:
        pass

    # export FN analysis same as v6
    try:
        test_user_ids = [all_nodes[i] for i in te_idx]
        test_results = pd.DataFrame({'user_id': test_user_ids, 'actual_status': y_te, 'pred_status': test_preds, 'pred_prob': test_probs})
        fn_users = test_results[(test_results['actual_status'] == 1) & (test_results['pred_status'] == 0)]
        features_all = final_df.reset_index()
        fn_analysis = fn_users.merge(features_all, on='user_id', how='left')
        fn_analysis.to_csv('gcn_v7_missed_blacklists_analysis.csv', index=False)
        print(f"\n已找出 {len(fn_analysis)} 個漏網之魚，詳細資料已存入 gcn_v7_missed_blacklists_analysis.csv")
        # 產生特徵分佈比較圖（Blacklist vs Normal）
        try:
            plot_feature_distribution(features_all, target_col='status', out_fname='feature_distribution_analysis.png')
        except Exception as e:
            print('Warning: feature distribution plot failed:', e)
        # --- 分析 usdt_buy_ratio == 1.0 的用戶關聯性 ---
        try:
            def analyze_usdt_buy1(df_all, G, out_csv='usdt_buy1_analysis.csv', out_png='usdt_buy1_network.png'):
                if 'usdt_buy_ratio' not in df_all.columns:
                    print('usdt_buy_ratio not in features; skipping usdt_buy1 analysis')
                    return
                # filter strict ==1 or >=0.999
                sel = df_all[df_all['usdt_buy_ratio'].astype(float) >= 0.999]
                if sel.shape[0] == 0:
                    print('No users with usdt_buy_ratio ~= 1.0 found')
                    return
                # summary stats
                cols_of_interest = ['user_id','status','is_isolated','shared_ip_degree','night_tx_ratio','dormant_intensity','is_round_amount_ratio']
                present = [c for c in cols_of_interest if c in df_all.columns]
                summary = sel[present].copy()
                summary.to_csv(out_csv, index=False)
                print(f'usdt_buy1: saved {len(summary)} rows to {out_csv}')

                # build small network: the selected nodes plus their neighbors
                nodes_sel = set(summary['user_id'].astype(int).tolist())
                sub_nodes = set(nodes_sel)
                for n in list(nodes_sel):
                    if n in G:
                        for nb in G.neighbors(n):
                            sub_nodes.add(nb)
                subG = G.subgraph(sub_nodes).copy()
                try:
                    pos = nx.spring_layout(subG, seed=42)
                    plt.figure(figsize=(10, 10))
                    node_colors = []
                    for n in subG.nodes():
                        if n in nodes_sel:
                            node_colors.append('red')
                        else:
                            node_colors.append('lightgrey')
                    nx.draw_networkx_nodes(subG, pos, node_size=80, node_color=node_colors, alpha=0.9)
                    nx.draw_networkx_edges(subG, pos, alpha=0.2)
                    plt.title(f'usdt_buy_ratio==1.0 network (selected red; n={len(nodes_sel)})')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(out_png, dpi=150)
                    print(f'Saved network view to {out_png}')
                except Exception as e:
                    print('Failed to draw usdt_buy1 network:', e)

            analyze_usdt_buy1(features_all, G)
        except Exception as e:
            print('Warning: usdt_buy1 analysis failed:', e)
    except Exception as e:
        print('Warning: FN export failed:', e)

    # --- t-SNE visualization of GAT embeddings, highlight FN in yellow ---
    try:
        # build FN set
        try:
            FN_ids = set(fn_analysis['user_id'].astype(int).tolist())
        except Exception:
            FN_ids = set(fn_users['user_id'].astype(int).tolist()) if 'fn_users' in locals() else set()

        latent = emb_scaled if 'emb_scaled' in locals() else emb
        sample_idx = np.arange(latent.shape[0])
        if latent.shape[0] > 5000:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(latent.shape[0], size=5000, replace=False)

        res = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(latent[sample_idx])
        labels_tsne = ['Blacklist' if int(y_np[i]) == 1 else 'Normal' for i in sample_idx]
        # convert to 'Missed' if in FN
        labels_tsne = ['Missed' if all_nodes[i] in FN_ids else lbl for i, lbl in zip(sample_idx, labels_tsne)]

        scatter_df = pd.DataFrame({'x': res[:, 0], 'y': res[:, 1], 'label': labels_tsne})
        plt.figure(figsize=(10, 8))

        # Plot in explicit order: Normal (low alpha) -> Blacklist (on top) -> Missed (largest, topmost)
        df_normal = scatter_df[scatter_df['label'] == 'Normal']
        df_black = scatter_df[scatter_df['label'] == 'Blacklist']
        df_missed = scatter_df[scatter_df['label'] == 'Missed']

        # Normal points: small, transparent, low zorder
        if len(df_normal) > 0:
            plt.scatter(df_normal['x'], df_normal['y'], c='skyblue', s=18, alpha=0.25, linewidths=0, zorder=1, marker='o')

        # Blacklist true positives: red, visible, drawn after normals
        if len(df_black) > 0:
            plt.scatter(df_black['x'], df_black['y'], c='red', s=70, alpha=0.95, edgecolors='black', linewidths=0.4, zorder=3, marker='o')

        # Missed (FN): yellow, larger, black edge, highest zorder so they stand out
        if len(df_missed) > 0:
            plt.scatter(df_missed['x'], df_missed['y'], c='yellow', s=110, alpha=1.0, edgecolors='black', linewidths=0.6, zorder=4, marker='o')

        # Build legend only for existing labels
        import matplotlib.patches as mpatches
        handles = []
        if len(df_missed) > 0:
            handles.append(mpatches.Patch(color='yellow', label=f'Missed (n={len(df_missed)})'))
        if len(df_black) > 0:
            handles.append(mpatches.Patch(color='red', label=f'Blacklist (n={len(df_black)})'))
        if len(df_normal) > 0:
            handles.append(mpatches.Patch(color='skyblue', label=f'Normal (n={len(df_normal)})'))
        if len(handles) > 0:
            plt.legend(handles=handles)

        plt.title('gcn_v7 t-SNE (yellow = missed FN; red = true blacklist)')
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.savefig('gcn_v7_tsne.png', dpi=150)
        print('Saved gcn_v7_tsne.png')
    except Exception as e:
        print('Warning: failed to generate t-SNE visualization:', e)

    print('Done v7.')
