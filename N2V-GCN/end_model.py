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
    behavior_df = pd.read_csv('all_features_analysis.csv')
    n2v_df = pd.read_csv('user_n2v.embeddings', sep=' ', skiprows=1, header=None)
    n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]
    final_df = behavior_df.merge(n2v_df, on='user_id', how='left')

    # amount transforms
    amount_cols = [c for c in final_df.columns if 'amount' in c.lower() or 'total' in c.lower()]
    for c in amount_cols:
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0.0)
        final_df[c] = np.log1p(final_df[c])

    # PCA on n2v -> reduce to 16, but drop first 5 dims if noisy
    # 針對沒轉帳、沒 n2v 的人補 0
    n2v_cols = [c for c in final_df.columns if c.startswith('n2v_')]
    final_df[n2v_cols] = final_df[n2v_cols].fillna(0)
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
    
    # 【修改 3】處理 status：保留 NaN，不強制填 0
    # 這樣我們才能區分「訓練集 (0/1)」與「預測集 (NaN)」
    if 'status' in final_df.columns:
        final_df['status'] = pd.to_numeric(final_df['status'], errors='coerce')

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
    # 強制所有特徵欄位數值化
    feat_cols = [c for c in final_df.columns if c != 'status']
    for col in feat_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

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

    # 【修改 4】建立 X 與 y，將 NaN 標籤標記為 -1
    X = np.zeros((len(all_nodes), len(feat_cols)), dtype=float)
    y = np.full(len(all_nodes), -1, dtype=int) # 預設全為 -1 (未標記)
    for i, node in enumerate(all_nodes):
        if node in final_df.index:
            row = final_df.loc[node]
            X[i, :] = row[feat_cols].values
            s = row.get('status', np.nan)
            if pd.isna(s):
                y[i] = -1
            else:
                y[i] = int(s)
        else:
            X[i, :] = 0.0
            y[i] = -1

    feature_names = feat_cols
    X = pd.DataFrame(X, index=all_nodes, columns=feature_names)
    y = pd.Series(y, index=all_nodes)
    return final_df, pd.DataFrame(X, index=all_nodes, columns=feat_cols), pd.Series(y, index=all_nodes), feat_cols

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
    print('Building graph and features (v8 Final Fixed)...')
    # 1. 先拿到基礎圖數據
    G_raw, _, _, graph_nodes = build_graph_data()
    
    # 2. 讀取官方預測名單
    pred_label_path = '../data/predict_label.csv' if os.path.exists('../data/predict_label.csv') else 'data/predict_label.csv'
    pred_label_df = pd.read_csv(pred_label_path)
    if 'user_id' not in pred_label_df.columns: pred_label_df.columns = ['user_id']
    submission_ids = pred_label_df['user_id'].astype(int).tolist()
    
    # 【關鍵修正：強迫把所有預測 ID 加進圖中，避免失蹤】
    G_raw.add_nodes_from(submission_ids)
    
    # 3. 定義全量節點表 (包含圖節點、預測名單、可能有的虛擬節點)
    all_nodes = sorted(list(G_raw.nodes()))
    id_map = {n: i for i, n in enumerate(all_nodes)}
    
    # 4. 載入特徵
    first_tx_map, round_ratio_map = optimized_feature_loader(all_nodes)
    final_df, X_df, y_series, feature_names = prepare_features_df_v8(id_map, all_nodes, G_raw, first_tx_map, round_ratio_map)

    # 5. 虛擬節點處理 (Hub-and-spoke)
    pure_buyers = [int(u) for u, v in final_df['is_pure_buyer'].items() if int(v) == 1]
    if len(pure_buyers) > 0:
        max_id = max([n for n in all_nodes if isinstance(n, (int, np.integer))])
        v_hub_id = int(max_id) + 1
        G_raw.add_node(v_hub_id)
        for u in pure_buyers:
            G_raw.add_edge(u, v_hub_id, virtual=1)
        
        # 再次更新總名單 (此時包含虛擬節點了)
        all_nodes = sorted(list(G_raw.nodes()))
        id_map = {n: i for i, n in enumerate(all_nodes)}

    # 6. 重新建立邊索引
    edge_list = []
    for u, v in G_raw.edges():
        edge_list.append([id_map[u], id_map[v]])
        edge_list.append([id_map[v], id_map[u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 7. 特徵與標籤對齊
    X_df = X_df.reindex(all_nodes).fillna(0.0)
    y_series = y_series.reindex(all_nodes).fillna(-1).astype(int)
    y_np = y_series.values

    # 8. 分割與 Mask 處理 (此處省略部分重複代碼，確保你的 tr_idx, val_idx 是從 labeled 中選取的)
    labeled_idx = np.where((y_np == 0) | (y_np == 1))[0]
    tr_idx, temp_idx = train_test_split(labeled_idx, stratify=y_np[labeled_idx], test_size=0.3, random_state=42)
    val_idx, te_idx = train_test_split(temp_idx, stratify=y_np[temp_idx], test_size=0.5, random_state=42)
    # 建立 Mask
    train_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    train_mask[tr_idx] = True; val_mask[val_idx] = True; test_mask[te_idx] = True

    # 轉為 Tensor
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_np, dtype=torch.long)
    y_tensor[y_tensor == -1] = 0 # GNN 訓練時標籤不能有 -1，但我們透過 Mask 隔離了它們

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = train_mask; data.val_mask = val_mask; data.test_mask = test_mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 8. 訓練 GAT
    model = EnhancedGAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    model = train_with_earlystop(model, data, F.cross_entropy, optimizer, patience=30, max_epochs=200)

    # 9. 產出 Embedding 並 Stacking
    model.eval()
    with torch.no_grad():
        emb = F.elu(model.conv1(data.x, data.edge_index)).cpu().numpy()
    
    is_isolated = (X_df['is_isolated'].values == 1)
    emb_scaled = emb * np.where(is_isolated[:, None], 0.05, 1.0)
    X_stack = np.hstack([emb_scaled, X_scaled])

    # 10. 訓練最後的分類器
    clf = HistGradientBoostingClassifier(max_iter=300)
    clf.fit(X_stack[tr_idx], y_np[tr_idx], sample_weight=np.where(y_np[tr_idx] == 1, 5.0, 1.0))

    # --------------------------------------------------
    # 最終預測結果輸出 (確保 12753 筆)
    # --------------------------------------------------
    try:
        # 找出 submission_ids 在總矩陣中的索引位置
        final_pred_indices = [id_map[uid] for uid in submission_ids]
        
        pred_X = X_stack[final_pred_indices]
        pred_probs = clf.predict_proba(pred_X)[:, 1]
        
        # 修正：如果 1 的數量太少，這裡使用動態門檻 (抓機率最高的前 4%)
        # 或者你也可以沿用 best_t (如下行)
        val_probs = clf.predict_proba(X_stack[val_idx])[:, 1]
        best_t, _ = find_best_threshold_by_f1(val_probs, y_np[val_idx])
        
        # 建議使用動態門檻，確保 Recall 夠高
        # target_count = int(len(submission_ids) * 0.04) # 抓 4%
        # dynamic_t = np.sort(pred_probs)[-target_count]
        pred_labels = (pred_probs >= best_t).astype(int)
        
        pred_result_df = pd.DataFrame({
            'user_id': submission_ids,
            'status': pred_labels
        })
        
        pred_result_df.to_csv('predict_result.csv', index=False)
        print(f"✅ 成功輸出！總行數: {len(pred_result_df)}")
        print(f"黑名單(1)數量: {pred_result_df['status'].sum()}")
        
    except Exception as e:
        print('預測結果產生失敗:', e)