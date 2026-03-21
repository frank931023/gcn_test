import os
import copy
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# v8: add is_pure_buyer / is_high_buy_ratio / is_night_owl, binning, virtual edges for pure buyers,
#      sample-weighting for imbalanced training, remove low-contribution PCA dims


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
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if len(edge_list) > 0 else torch.empty((2,0), dtype=torch.long)
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


def optimized_feature_loader(user_ids, tx_files=None, usecols=None, chunk_size=100000):
    """一次性讀取需要的交易欄位，並回傳 first_tx_map 與 round_ratio_map。
    目的：避免多次讀取大型 CSV，並且只讀取必要欄位。
    """
    if tx_files is None:
        tx_files = ['../data/crypto_transfer.csv','../data/twd_transfer.csv','../data/usdt_twd_trading.csv','../data/usdt_swap.csv']
    first_times = {u: pd.NaT for u in user_ids}
    counts = {u: 0 for u in user_ids}
    round_counts = {u: 0 for u in user_ids}

    for f in tx_files:
        if not os.path.exists(f):
            continue
        # 尝试只读取必要欄位以節省記憶體
        try:
            # 若檔案很大，可使用 chunksize
            for chunk in pd.read_csv(f, usecols=None, chunksize=chunk_size):
                # 推斷 user 與 時間/amount 欄位
                cols = chunk.columns.tolist()
                date_cols = [c for c in cols if 'time' in c.lower() or 'date' in c.lower() or 'created' in c.lower()]
                amt_cols = [c for c in cols if 'amount' in c.lower()]
                user_cols = [c for c in cols if 'user' in c.lower() and ('id' in c.lower() or 'user' in c.lower())]
                if len(user_cols) == 0:
                    continue
                # 使用第一個匹配的欄位
                uc = user_cols[0]
                dc = date_cols[0] if len(date_cols) > 0 else None
                ac = amt_cols[0] if len(amt_cols) > 0 else None
                tmp = chunk[[uc] + ([dc] if dc is not None else []) + ([ac] if ac is not None else [])].dropna(subset=[uc])
                if dc is not None:
                    try:
                        tmp[dc] = try_parse_datetime_series(tmp[dc])
                    except Exception:
                        pass
                for uid, grp in tmp.groupby(uc):
                    try:
                        uid_i = int(uid)
                    except Exception:
                        continue
                    if uid_i not in counts:
                        continue
                    # first tx
                    if dc is not None:
                        tmin = grp[dc].min()
                        if pd.isna(first_times[uid_i]) or (not pd.isna(tmin) and tmin < first_times[uid_i]):
                            first_times[uid_i] = tmin
                    # round amount ratio
                    if ac is not None:
                        vals = pd.to_numeric(grp[ac], errors='coerce').dropna().astype(float)
                        if len(vals) > 0:
                            counts[uid_i] += len(vals)
                            # 判定是否為整數/大整數（示例：模 10000）
                            round_counts[uid_i] += (np.isclose(vals % 10000, 0)).sum()
        except ValueError:
            # 若檔案不支援 chunksize（小檔案），直接讀取
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            cols = df.columns.tolist()
            date_cols = [c for c in cols if 'time' in c.lower() or 'date' in c.lower() or 'created' in c.lower()]
            amt_cols = [c for c in cols if 'amount' in c.lower()]
            user_cols = [c for c in cols if 'user' in c.lower() and ('id' in c.lower() or 'user' in c.lower())]
            if len(user_cols) == 0:
                continue
            uc = user_cols[0]
            dc = date_cols[0] if len(date_cols) > 0 else None
            ac = amt_cols[0] if len(amt_cols) > 0 else None
            tmp = df[[uc] + ([dc] if dc is not None else []) + ([ac] if ac is not None else [])].dropna(subset=[uc])
            if dc is not None:
                try:
                    tmp[dc] = try_parse_datetime_series(tmp[dc])
                except Exception:
                    pass
            for uid, grp in tmp.groupby(uc):
                try:
                    uid_i = int(uid)
                except Exception:
                    continue
                if uid_i not in counts:
                    continue
                if dc is not None:
                    tmin = grp[dc].min()
                    if pd.isna(first_times[uid_i]) or (not pd.isna(tmin) and tmin < first_times[uid_i]):
                        first_times[uid_i] = tmin
                if ac is not None:
                    vals = pd.to_numeric(grp[ac], errors='coerce').dropna().astype(float)
                    if len(vals) > 0:
                        counts[uid_i] += len(vals)
                        round_counts[uid_i] += (np.isclose(vals % 10000, 0)).sum()

    round_ratio_map = {u: (round_counts[u] / counts[u]) if counts[u] > 0 else 0.0 for u in user_ids}
    return first_times, round_ratio_map


def prepare_features_df_v8(id_map, all_nodes, G, first_tx_map=None, round_ratio_map=None):
    behavior_df = pd.read_csv('extended_features_analysis.csv')
    n2v_df = pd.read_csv('user_n2v.embeddings', sep=' ', skiprows=1, header=None)
    n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]
    final_df = behavior_df.merge(n2v_df, on='user_id', how='inner')

    # amount transforms
    amount_cols = [c for c in final_df.columns if 'amount' in c.lower() or 'total' in c.lower()]
    for c in amount_cols:
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0.0)
        final_df[c] = np.log1p(final_df[c])

    # PCA on n2v -> reduce to 16, but drop first 5 dims if noisy
    n2v_cols = [c for c in final_df.columns if c.startswith('n2v_')]
    if len(n2v_cols) > 0:
        pca = PCA(n_components=16, random_state=42)
        n2v_vals = final_df[n2v_cols].fillna(0).values
        n2v_red = pca.fit_transform(n2v_vals)
        for i in range(n2v_red.shape[1]):
            final_df[f'n2v_pca_{i}'] = n2v_red[:, i]
        final_df = final_df.drop(columns=n2v_cols)
        # drop first few PCA dims that often overlap with normal users
        drop_pca = [f'n2v_pca_{i}' for i in range(5)]
        for d in drop_pca:
            if d in final_df.columns:
                final_df.drop(columns=[d], inplace=True)

    final_df = final_df.set_index('user_id')

    # neighbor risk and neighbor avg
    status_dict = final_df['status'].to_dict() if 'status' in final_df.columns else {}
    neigh_risk = {}
    neigh_n2v_avg = {}
    degree_map = {}
    for node in all_nodes:
        degree_map[node] = G.degree(node) if node in G else 0
        if node not in final_df.index:
            neigh_risk[node] = 0.0
            neigh_n2v_avg[node] = np.zeros(max(1, len([c for c in final_df.columns if c.startswith('n2v_pca_')])) )
            continue
        neighbors = list(G.neighbors(node)) if node in G else []
        if len(neighbors) == 0:
            neigh_risk[node] = 0.0
            neigh_n2v_avg[node] = np.zeros(max(1, len([c for c in final_df.columns if c.startswith('n2v_pca_')])) )
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
            neigh_n2v_avg[node] = np.zeros(max(1, len([c for c in final_df.columns if c.startswith('n2v_pca_')])) )

    final_df['neighbor_risk_score'] = pd.Series(neigh_risk)
    # add neighbor n2v avg columns
    n2v_len = len([c for c in final_df.columns if c.startswith('n2v_pca_')])
    for i in range(n2v_len):
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

    # first_tx delay
    user_ids = list(final_df.index)
    if first_tx_map is None:
        first_tx = find_first_tx_times(user_ids)
    else:
        first_tx = first_tx_map
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
    if round_ratio_map is None:
        round_ratios = compute_round_amount_ratio(user_ids)
    else:
        round_ratios = round_ratio_map
    final_df['is_round_amount_ratio'] = pd.Series(round_ratios)

    # dormant intensity
    if 'swap_twd_avg' in final_df.columns:
        final_df['dormant_intensity'] = final_df['first_tx_delay_days'] * final_df['swap_twd_avg']
    else:
        final_df['dormant_intensity'] = final_df['first_tx_delay_days'] * 0.0

    # is_isolated
    final_df['shared_ip_degree'] = pd.Series(degree_map)
    final_df['is_isolated'] = (final_df['shared_ip_degree'] < 2).astype(int)

    # new v8 features
    final_df['is_pure_buyer'] = (final_df['usdt_buy_ratio'].astype(float) >= 0.999).astype(int) if 'usdt_buy_ratio' in final_df.columns else 0
    final_df['is_high_buy_ratio'] = (final_df['usdt_buy_ratio'].astype(float) > 0.95).astype(int) if 'usdt_buy_ratio' in final_df.columns else 0
    final_df['is_night_owl'] = (final_df['night_tx_ratio'].astype(float) > 0.7).astype(int) if 'night_tx_ratio' in final_df.columns else 0

    # binning night_tx_ratio and dormant_intensity
    if 'night_tx_ratio' in final_df.columns:
        final_df['night_bin_high'] = (final_df['night_tx_ratio'].astype(float) > 0.8).astype(int)
    else:
        final_df['night_bin_high'] = 0
    try:
        final_df['dormant_bin'] = pd.cut(final_df['dormant_intensity'].replace([np.inf, -np.inf], np.nan).fillna(0), bins=[-1,0,1,7,30,1e9], labels=False)
    except Exception:
        final_df['dormant_bin'] = 0

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

    # select features -> drop low-importance PCA dims if present
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


if __name__ == '__main__':
    print('Building graph and features (v8 memory-optimized)...')
    G, id_map, edge_index, all_nodes = build_graph_data()
    # 一次性讀取交易檔以計算 first_tx 與 round_amount_ratio，避免重複讀檔
    user_ids_for_loader = list(set(all_nodes))
    first_tx_map, round_ratio_map = optimized_feature_loader(user_ids_for_loader)
    final_df, X_df, y_series, feature_names = prepare_features_df_v8(id_map, all_nodes, G, first_tx_map, round_ratio_map)

    # 虛擬節點策略 (hub-and-spoke): 對 pure_buyers 建立一個虛擬中心節點，避免 O(N^2) 邊
    pure_buyers = [int(u) for u, v in final_df['is_pure_buyer'].items() if int(v) == 1 and u in G]
    if len(pure_buyers) > 0:
        try:
            max_id = max([n for n in all_nodes if isinstance(n, (int, np.integer))])
            v_hub_id = int(max_id) + 1
        except Exception:
            v_hub_id = 999999999
        # add hub node and connect spokes
        if not G.has_node(v_hub_id):
            G.add_node(v_hub_id)
        for u in pure_buyers:
            if u in G and not G.has_edge(u, v_hub_id):
                G.add_edge(u, v_hub_id, virtual=1)

    # rebuild edge_index allowing new edges (much smaller now)
    all_nodes = sorted(list(G.nodes()))
    id_map = {n: i for i, n in enumerate(all_nodes)}
    edge_list = []
    for u, v in G.edges():
        edge_list.append([id_map[u], id_map[v]])
        edge_list.append([id_map[v], id_map[u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if len(edge_list)>0 else torch.empty((2,0), dtype=torch.long)

    # 確保 X_df / y_series 包含所有節點（包含剛建立的 hub），缺失節點以 0 向量填補
    try:
        missing_nodes = [n for n in all_nodes if n not in X_df.index]
        if len(missing_nodes) > 0:
            print(f"Adding {len(missing_nodes)} missing nodes to feature matrix (filled with zeros)")
            zeros = np.zeros((len(missing_nodes), X_df.shape[1]), dtype=float)
            df_missing = pd.DataFrame(zeros, index=missing_nodes, columns=X_df.columns)
            X_df = pd.concat([X_df, df_missing])
            y_missing = pd.Series([0] * len(missing_nodes), index=missing_nodes)
            y_series = pd.concat([y_series, y_missing])
        # reindex to exact all_nodes order
        X_df = X_df.reindex(all_nodes).fillna(0.0)
        y_series = y_series.reindex(all_nodes).fillna(0).astype(int)
    except Exception as e:
        print('Warning: failed to align X_df/y_series with new all_nodes:', e)

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

    print('Training GAT encoder (v8)...')
    model = train_with_earlystop(model, data, criterion, optimizer, patience=30, max_epochs=200)

    model.eval()
    with torch.no_grad():
        emb = F.elu(model.conv1(data.x, data.edge_index)).cpu().numpy()

    # behavioral fallback: scale isolated embeddings more strongly
    is_isolated = X_df['is_isolated'].values if 'is_isolated' in X_df.columns else np.zeros(len(X_df))
    scale_f = np.where(is_isolated == 1, 0.05, 1.0)
    emb_scaled = emb * scale_f[:, None]

    # stacking: embeddings (scaled) + raw features
    X_stack = np.hstack([emb_scaled, X_scaled])
    X_tr = X_stack[tr_idx]; y_tr = y_np[tr_idx]
    X_val = X_stack[val_idx]; y_val = y_np[val_idx]
    X_te = X_stack[te_idx]; y_te = y_np[te_idx]

    # classifier with sample weights to upweight minority class
    clf = HistGradientBoostingClassifier(max_iter=300)
    sample_weight = np.where(y_tr == 1, 5.0, 1.0)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)

    val_probs = clf.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold_by_f1(val_probs, y_val)
    print(f'Best threshold on val by F1: {best_t:.4f}, val F1: {best_f1:.4f}')

    test_probs = clf.predict_proba(X_te)[:, 1]
    test_preds = (test_probs >= best_t).astype(int)
    cm = confusion_matrix(y_te, test_preds)
    print('\nStacked model Test Confusion Matrix:\n', cm)
    print('\nStacked Classification Report:')
    print(classification_report(y_te, test_preds))

    # --------------------------------------------------
    # SHAP explainability (summary + per-FN contribution)
    # --------------------------------------------------
    try:
        import shap
        print('Computing SHAP values (this may take some time)...')
        # 建立 feature name list for stacking input
        emb_dim = emb_scaled.shape[1] if 'emb_scaled' in locals() else emb.shape[1]
        stack_feature_names = [f'emb_{i}' for i in range(emb_dim)] + list(feature_names)

        # 使用 TreeExplainer，對 HistGradientBoostingClassifier 及類似樹模型有效
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values_raw = explainer.shap_values(X_te)
        except Exception:
            # fallback to generic Explainer
            explainer = shap.Explainer(clf, X_tr)
            shap_exp = explainer(X_te)
            # shap_exp.values 可能是 (n_samples, n_features) 或 (n_samples, n_outputs, n_features)
            shap_values_raw = getattr(shap_exp, 'values', None)

        # normalize shap_values to numpy array for positive class
        if isinstance(shap_values_raw, list):
            # list of arrays per class -> take class 1
            sv_pos = np.array(shap_values_raw[1]) if len(shap_values_raw) > 1 else np.array(shap_values_raw[0])
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3:
                # shape (n_classes, n_samples, n_features) or (n_samples, n_classes, n_features)
                if shap_values_raw.shape[0] == 2:
                    sv_pos = shap_values_raw[1]
                else:
                    sv_pos = shap_values_raw[:, 1, :]
            else:
                sv_pos = shap_values_raw
        else:
            sv_pos = None

        if sv_pos is not None:
            mean_abs = np.mean(np.abs(sv_pos), axis=0)
            shap_df = pd.DataFrame({'feature': stack_feature_names, 'mean_abs_shap': mean_abs})
            shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
            shap_df.to_csv('gcn_v8_shap_feature_importance.csv', index=False)

            # plot top features
            try:
                topk = min(30, len(shap_df))
                plt.figure(figsize=(8, max(6, topk*0.2)))
                plt.barh(shap_df['feature'].iloc[:topk][::-1], shap_df['mean_abs_shap'].iloc[:topk][::-1])
                plt.xlabel('mean |SHAP value|')
                plt.title('SHAP feature importance (stacked model)')
                plt.tight_layout()
                plt.savefig('gcn_v8_shap_summary.png', dpi=150)
                print('Saved gcn_v8_shap_summary.png and gcn_v8_shap_feature_importance.csv')
            except Exception as e:
                print('Warning: failed to plot SHAP summary:', e)

            # output per-FN explanations (top 3) as CSV and small bar plots
            try:
                fn_user_ids = fn_users['user_id'].astype(int).tolist() if 'fn_users' in locals() else []
                saved = 0
                for uid in fn_user_ids[:3]:
                    if uid not in all_nodes:
                        continue
                    idx_node = all_nodes.index(uid)
                    # find position in test set
                    pos = np.where(te_idx == idx_node)[0]
                    if len(pos) == 0:
                        continue
                    samp = pos[0]
                    contrib = sv_pos[samp]
                    df_contrib = pd.DataFrame({'feature': stack_feature_names, 'shap': contrib})
                    df_contrib['abs_shap'] = df_contrib['shap'].abs()
                    df_contrib = df_contrib.sort_values('abs_shap', ascending=False)
                    fname_csv = f'gcn_v8_shap_fn_{uid}.csv'
                    df_contrib.to_csv(fname_csv, index=False)
                    # bar plot top 20
                    try:
                        topn = min(20, len(df_contrib))
                        plt.figure(figsize=(8, max(4, topn*0.2)))
                        plt.barh(df_contrib['feature'].iloc[:topn][::-1], df_contrib['shap'].iloc[:topn][::-1], color='orange')
                        plt.title(f'SHAP contributions for FN user {uid}')
                        plt.tight_layout()
                        plt.savefig(f'gcn_v8_shap_fn_{uid}.png', dpi=150)
                    except Exception:
                        pass
                    saved += 1
                if saved > 0:
                    print(f'Saved per-FN SHAP outputs for {saved} users')
            except Exception as e:
                print('Warning: failed to write per-FN SHAP outputs:', e)
        else:
            print('SHAP: could not compute shap array (unexpected format)')
    except Exception as e:
        print('SHAP not available or failed:', e)
    # save artifacts
    try:
        pd.Series([best_t], index=['best_threshold']).to_csv('gcn_v8_best_threshold.csv')
        torch.save(model.state_dict(), 'gcn_v8_gat.pt')
    except Exception:
        pass

    # export FN
    try:
        test_user_ids = [all_nodes[i] for i in te_idx]
        test_results = pd.DataFrame({'user_id': test_user_ids, 'actual_status': y_te, 'pred_status': test_preds, 'pred_prob': test_probs})
        fn_users = test_results[(test_results['actual_status'] == 1) & (test_results['pred_status'] == 0)]
        features_all = final_df.reset_index()
        fn_analysis = fn_users.merge(features_all, on='user_id', how='left')
        fn_analysis.to_csv('gcn_v8_missed_blacklists_analysis.csv', index=False)
        print(f"\n已找出 {len(fn_analysis)} 個漏網之魚，詳細資料已存入 gcn_v8_missed_blacklists_analysis.csv")
    except Exception as e:
        print('Warning: FN export failed:', e)

    # graph view
    try:
        FN_ids = set(fn_analysis['user_id'].astype(int).tolist()) if 'fn_analysis' in locals() else set()
        # 若圖太大，抽樣子圖以降低 spring_layout / 繪圖成本
        max_plot_nodes = 2000
        G_plot = G
        if G.number_of_nodes() > max_plot_nodes:
            rng = np.random.RandomState(42)
            sampled_nodes = list(rng.choice(list(G.nodes()), size=max_plot_nodes, replace=False))
            G_plot = G.subgraph(sampled_nodes).copy()
            print(f"Graph has {G.number_of_nodes()} nodes; plotting sampled subgraph of {G_plot.number_of_nodes()} nodes")
        pos = nx.spring_layout(G_plot, seed=42)
        node_colors = []
        for n in G_plot.nodes():
            if n in FN_ids:
                node_colors.append('yellow')
            elif n in pure_buyers:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(G_plot, pos, node_size=30, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G_plot, pos, alpha=0.15)
        plt.title('Graph v8 — red=pure_buyers (hub), yellow=missed')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('gcn_v8_graph.png', dpi=200)
        print('Saved gcn_v8_graph.png')
    except Exception as e:
        print('Warning: failed to generate graph visualization:', e)

    # t-SNE
    try:
        latent = emb_scaled if 'emb_scaled' in locals() else emb
        sample_idx = np.arange(latent.shape[0])
        if latent.shape[0] > 5000:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(latent.shape[0], size=5000, replace=False)
        res = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(latent[sample_idx])
        labels_tsne = ['Blacklist' if int(y_np[i]) == 1 else 'Normal' for i in sample_idx]
        labels_tsne = ['Missed' if all_nodes[i] in FN_ids else lbl for i, lbl in zip(sample_idx, labels_tsne)]
        scatter_df = pd.DataFrame({'x': res[:,0], 'y': res[:,1], 'label': labels_tsne})
        plt.figure(figsize=(10,8))
        df_normal = scatter_df[scatter_df['label']=='Normal']
        df_black = scatter_df[scatter_df['label']=='Blacklist']
        df_missed = scatter_df[scatter_df['label']=='Missed']
        if len(df_normal)>0:
            plt.scatter(df_normal['x'], df_normal['y'], c='skyblue', s=18, alpha=0.2, zorder=1)
        if len(df_black)>0:
            plt.scatter(df_black['x'], df_black['y'], c='red', s=70, alpha=0.9, edgecolors='black', zorder=3)
        if len(df_missed)>0:
            plt.scatter(df_missed['x'], df_missed['y'], c='yellow', s=120, alpha=1.0, edgecolors='black', zorder=4)
        import matplotlib.patches as mpatches
        handles = []
        if len(df_missed)>0:
            handles.append(mpatches.Patch(color='yellow', label=f'Missed (n={len(df_missed)})'))
        if len(df_black)>0:
            handles.append(mpatches.Patch(color='red', label=f'Blacklist (n={len(df_black)})'))
        if len(df_normal)>0:
            handles.append(mpatches.Patch(color='skyblue', label=f'Normal (n={len(df_normal)})'))
        if len(handles)>0:
            plt.legend(handles=handles)
        plt.title('gcn_v8 t-SNE')
        plt.tight_layout()
        plt.savefig('gcn_v8_tsne.png', dpi=150)
        print('Saved gcn_v8_tsne.png')
    except Exception as e:
        print('Warning: failed to generate t-SNE visualization:', e)

    print('Done v8.')
