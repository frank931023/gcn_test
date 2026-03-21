import os
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch_geometric.nn import GATv2Conv

# Reuse processing and model utilities from gcn_v8 when available
try:
    from gcn_v8 import prepare_features_df_v8, optimized_feature_loader, EnhancedGAT, train_with_earlystop, find_best_threshold_by_f1
except Exception:
    # if import fails (running as script from different cwd), try relative import
    try:
        from .gcn_v8 import prepare_features_df_v8, optimized_feature_loader, EnhancedGAT, train_with_earlystop, find_best_threshold_by_f1
    except Exception:
        prepare_features_df_v8 = None
        optimized_feature_loader = None
        EnhancedGAT = None
        train_with_earlystop = None
        find_best_threshold_by_f1 = None


def build_hetero_graph():
    """建立包含 user/ip/wallet 節點的異質（實作為同質 ID with type prefix）圖。
    - user nodes: 'user:<id>'
    - ip nodes: 'ip:<ipstr>'  (if available in behavior features)
    - wallet nodes: 'wallet:<addr>' (從交易檔萃取)
    """
    G = nx.Graph()
    # add user-user edges from shared_ip_edges or relation edges if available
    shared_ip_f = '../data/edge/shared_ip_edges.csv'
    if os.path.exists(shared_ip_f):
        try:
            df = pd.read_csv(shared_ip_f)
            # try common column names
            u1 = 'user_id_1' if 'user_id_1' in df.columns else df.columns[0]
            u2 = 'user_id_2' if 'user_id_2' in df.columns else df.columns[1] if len(df.columns)>1 else None
            if u2 is not None:
                for _, r in df[[u1, u2]].dropna().iterrows():
                    a = f'user:{int(r[u1])}'; b = f'user:{int(r[u2])}'
                    G.add_node(a, ntype='user'); G.add_node(b, ntype='user')
                    G.add_edge(a, b)
        except Exception:
            pass

    # add transactions: connect user to wallet nodes
    tx_f = '../data/crypto_transfer.csv'
    if os.path.exists(tx_f):
        try:
            tx = pd.read_csv(tx_f, usecols=None)
            cols = tx.columns.tolist()
            user_cols = [c for c in cols if 'user' in c.lower() and ('id' in c.lower() or 'user' in c.lower())]
            wallet_cols = [c for c in cols if any(k in c.lower() for k in ['wallet', 'addr', 'address', 'hash'])]
            if len(user_cols) > 0 and len(wallet_cols) > 0:
                uc = user_cols[0]
                wc = wallet_cols[0]
                for _, r in tx[[uc, wc]].dropna().iterrows():
                    try:
                        uid = int(r[uc]); addr = str(r[wc])
                    except Exception:
                        continue
                    u_node = f'user:{uid}'; w_node = f'wallet:{addr}'
                    G.add_node(u_node, ntype='user'); G.add_node(w_node, ntype='wallet')
                    G.add_edge(u_node, w_node)
        except Exception:
            pass

    # try to add ip nodes from extended features if present
    beh_f = 'extended_features_analysis.csv'
    if os.path.exists(beh_f):
        try:
            beh = pd.read_csv(beh_f)
            ip_cols = [c for c in beh.columns if 'ip' in c.lower()]
            id_col = 'user_id' if 'user_id' in beh.columns else beh.columns[0]
            if len(ip_cols) > 0:
                ipc = ip_cols[0]
                for _, r in beh[[id_col, ipc]].dropna().iterrows():
                    try:
                        uid = int(r[id_col]); ip = str(r[ipc])
                    except Exception:
                        continue
                    u_node = f'user:{uid}'; ip_node = f'ip:{ip}'
                    G.add_node(u_node, ntype='user'); G.add_node(ip_node, ntype='ip')
                    G.add_edge(u_node, ip_node)
        except Exception:
            pass

    # Convert nodes to integer ids while keeping user ids unchanged.
    # Assign synthetic integer ids for wallets and ips starting after max user id.
    users = set()
    wallets = set()
    ips = set()
    for n in list(G.nodes()):
        attr = G.nodes[n].get('ntype', None)
        if attr == 'user' or (isinstance(n, str) and n.startswith('user:')):
            # node name may be string 'user:<id>' or int
            try:
                uid = int(str(n).split(':', 1)[1]) if isinstance(n, str) else int(n)
            except Exception:
                continue
            users.add(uid)
        elif attr == 'wallet' or (isinstance(n, str) and n.startswith('wallet:')):
            wallets.add(str(n))
        elif attr == 'ip' or (isinstance(n, str) and n.startswith('ip:')):
            ips.add(str(n))

    max_user = max(users) if len(users) > 0 else 0
    next_id = max_user + 1
    wallet_map = {w: next_id + i for i, w in enumerate(sorted(wallets))}
    next_id += len(wallets)
    ip_map = {ip: next_id + i for i, ip in enumerate(sorted(ips))}

    # build integer-id graph
    G2 = nx.Graph()
    for u, v in G.edges():
        def to_int(node):
            if isinstance(node, int):
                return int(node)
            s = str(node)
            if s.startswith('user:'):
                return int(s.split(':', 1)[1])
            if s.startswith('wallet:'):
                return wallet_map.get(s)
            if s.startswith('ip:'):
                return ip_map.get(s)
            # fallback
            try:
                return int(s)
            except Exception:
                return None
        a = to_int(u); b = to_int(v)
        if a is None or b is None:
            continue
        G2.add_node(a); G2.add_node(b)
        G2.add_edge(a, b)

    all_nodes = sorted(list(G2.nodes()))
    id_map = {n: i for i, n in enumerate(all_nodes)}
    edge_list = []
    for u, v in G2.edges():
        edge_list.append([id_map[u], id_map[v]])
        edge_list.append([id_map[v], id_map[u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if len(edge_list)>0 else torch.empty((2,0), dtype=torch.long)
    return G2, all_nodes, id_map, edge_index


def build_features_matrix(all_nodes):
    """為所有節點建立特徵矩陣：
    - user 節點填入 extended features + n2v PCA（如存在）
    - ip/wallet 節點填 0，並加入 node-type one-hot 與 degree later
    返回 X_df (DataFrame)、y_series (labels only for user nodes)
    """
    # load user features
    user_feat = pd.DataFrame()
    if os.path.exists('extended_features_analysis.csv'):
        try:
            user_feat = pd.read_csv('extended_features_analysis.csv').set_index('user_id')
        except Exception:
            user_feat = pd.DataFrame()
    # coerce object columns: try datetime -> timestamp, else numeric -> fill 0
    if not user_feat.empty:
        for col in user_feat.columns:
            if user_feat[col].dtype == object:
                # try parse datetime
                try:
                    dt = pd.to_datetime(user_feat[col], errors='coerce', utc=True)
                    if dt.notna().any():
                        user_feat[col] = (dt.astype('int64') // 10**9).fillna(0)
                        continue
                except Exception:
                    pass
                # fallback to numeric
                user_feat[col] = pd.to_numeric(user_feat[col], errors='coerce').fillna(0)
    # n2v embeddings optional
    if os.path.exists('user_n2v.embeddings') and not user_feat.empty:
        try:
            n2v = pd.read_csv('user_n2v.embeddings', sep=' ', skiprows=1, header=None)
            n2v.columns = ['user_id'] + [f'n2v_{i}' for i in range(n2v.shape[1]-1)]
            n2v = n2v.set_index('user_id')
            user_feat = user_feat.merge(n2v, left_index=True, right_index=True, how='left')
        except Exception:
            pass

    # determine user feature columns (exclude 'status' to avoid target leakage)
    user_cols = [c for c in list(user_feat.columns) if c != 'status'] if not user_feat.empty else []
    base_dim = max(1, len(user_cols))

    # prepare arrays
    X = np.zeros((len(all_nodes), base_dim + 3 + 1), dtype=float)  # user_feats + type_onehot(3) + degree
    y = np.zeros(len(all_nodes), dtype=int)

    node_type_map = {}
    # allow user nodes to be integers (from mapped hetero graph) or strings like 'user:<id>'
    for i, n in enumerate(all_nodes):
        node_type = 'wallet'
        uid = None
        # integer node -> assume user id
        if isinstance(n, (int, np.integer)):
            node_type = 'user'
            uid = int(n)
        else:
            s = str(n)
            if s.startswith('user:'):
                node_type = 'user'
                try:
                    uid = int(s.split(':', 1)[1])
                except Exception:
                    uid = None
            elif s.startswith('ip:'):
                node_type = 'ip'
            elif s.startswith('wallet:'):
                node_type = 'wallet'
            else:
                # fallback: if string looks numeric, treat as user
                if s.isdigit():
                    node_type = 'user'
                    uid = int(s)
        node_type_map[n] = node_type
        if node_type == 'user' and uid is not None and not user_feat.empty:
            # support user_feat index being int or str
            row = None
            try:
                if uid in user_feat.index:
                    row = user_feat.loc[uid, user_cols]
                elif str(uid) in user_feat.index:
                    row = user_feat.loc[str(uid), user_cols]
            except Exception:
                row = None
            if row is not None:
                vals = pd.Series(row).fillna(0).values.astype(float)
                X[i, :len(vals)] = vals
                # label
                if 'status' in user_feat.columns:
                    try:
                        if uid in user_feat.index:
                            y[i] = int(user_feat.loc[uid, 'status'])
                        else:
                            y[i] = int(user_feat.loc[str(uid), 'status'])
                    except Exception:
                        y[i] = 0
    # add type one-hot and degree
    G_tmp = nx.Graph()
    G_tmp.add_nodes_from(all_nodes)
    # degree will be filled by caller -- set zeros for now
    for i, n in enumerate(all_nodes):
        t = node_type_map.get(n, 'wallet')
        if t == 'user':
            X[i, base_dim + 0] = 1.0
        elif t == 'ip':
            X[i, base_dim + 1] = 1.0
        else:
            X[i, base_dim + 2] = 1.0

    X_df = pd.DataFrame(X, index=all_nodes, columns=[f'feat_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, index=all_nodes)
    return X_df, y_series, node_type_map


class SimpleGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # smaller model (fewer heads, smaller hidden) to reduce overfitting and speed up
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, dropout=0.3)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, dropout=0.3)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = F.elu(h1)
        h2 = self.conv2(h1, edge_index)
        combined = torch.cat([h2, x], dim=1)
        return self.classifier(combined)


if __name__ == '__main__':
    print('Building heterogeneous graph (v9)...')
    G, all_nodes, id_map, edge_index = build_hetero_graph()
    print('Nodes:', len(all_nodes), 'Edges:', G.number_of_edges())

    X_df, y_series, node_type_map = build_features_matrix(all_nodes)

    # rebuild degree into last column
    degs = np.array([G.degree(n) for n in all_nodes], dtype=float)
    X_df.iloc[:, -1] = degs

    # split only user nodes for stratified sampling
    user_positions = [i for i, n in enumerate(all_nodes) if node_type_map.get(n) == 'user']
    y_users = y_series.iloc[user_positions].values
    if len(user_positions) < 10 or y_users.sum() == 0:
        print('Not enough user labels to train; exiting.')
        raise SystemExit(0)

    tr_u, tmp_u = train_test_split(user_positions, stratify=y_users, test_size=0.3, random_state=42)
    val_u, te_u = train_test_split(tmp_u, stratify=y_series.iloc[tmp_u].values, test_size=0.5, random_state=42)

    train_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    train_mask[tr_u] = True; val_mask[val_u] = True; test_mask[te_u] = True

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_series.values, dtype=torch.long)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = train_mask; data.val_mask = val_mask; data.test_mask = test_mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = SimpleGAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = F.cross_entropy

    # simple training loop (reduced epochs + earlier stopping to avoid overfitting)
    best_val = 1e9; patience = 10; cnt = 0; best_state = None
    for epoch in range(1, 101):
        model.train(); optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(data.x, data.edge_index)
            val_loss = criterion(v_logits[data.val_mask], data.y[data.val_mask]).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; cnt = 0
        else:
            cnt += 1
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}')
        if cnt >= patience:
            print('Early stopping at', epoch); break
    if best_state is not None:
        model.load_state_dict(best_state)

    # get embeddings and stack for classifier only on users
    model.eval()
    with torch.no_grad():
        emb = F.elu(model.conv1(data.x, data.edge_index)).cpu().numpy()

    # stacking train classifier on user nodes
    X_stack = np.hstack([emb[user_positions], X_scaled[user_positions]])
    y_u = y_series.iloc[user_positions].values
    # map user_positions to train/val/test subset indices
    idx_map = {p: i for i, p in enumerate(user_positions)}
    tr_idx = [idx_map[p] for p in tr_u]
    val_idx = [idx_map[p] for p in val_u]
    te_idx = [idx_map[p] for p in te_u]

    clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    sample_weight = np.where(y_u[tr_idx] == 1, 5.0, 1.0)
    clf.fit(X_stack[tr_idx], y_u[tr_idx], sample_weight=sample_weight)

    val_probs = clf.predict_proba(X_stack[val_idx])[:, 1]
    # find best threshold
    best_f1 = -1; best_t = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        p = (val_probs >= t).astype(int)
        f = f1_score(y_u[val_idx], p, zero_division=0)
        if f > best_f1:
            best_f1 = f; best_t = t
    print('Best val threshold', best_t, 'f1', best_f1)

    te_probs = clf.predict_proba(X_stack[te_idx])[:, 1]
    te_preds = (te_probs >= best_t).astype(int)
    print('Test Confusion Matrix:')
    print(confusion_matrix(y_u[te_idx], te_preds))
    print(classification_report(y_u[te_idx], te_preds))

    print('v9 heterogeneous run complete.')
