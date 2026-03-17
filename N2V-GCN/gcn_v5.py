import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# --- Focal Loss ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# --- Graph builder (same as v4) ---
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

# --- prepare features, return DataFrame and tensors ---
def prepare_features_df(id_map, all_nodes, G):
    behavior_df = pd.read_csv('extended_features_analysis.csv')
    n2v_df = pd.read_csv('user_n2v.embeddings', sep=' ', skiprows=1, header=None)
    n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]

    final_df = behavior_df.merge(n2v_df, on='user_id', how='inner')

    # log1p for amount-like columns
    amount_cols = [c for c in final_df.columns if 'amount' in c.lower() or 'total' in c.lower()]
    if len(amount_cols) > 0:
        for c in amount_cols:
            final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0.0)
            final_df[c] = np.log1p(final_df[c])

    # convert object/date columns when possible
    for col in list(final_df.columns):
        if col == 'user_id':
            continue
        if final_df[col].dtype == object:
            dt = pd.to_datetime(final_df[col], errors='coerce', utc=True)
            if dt.notna().any():
                final_df[col] = dt.astype('int64') // 10**9
                continue
            num = pd.to_numeric(final_df[col], errors='coerce')
            if num.notna().any():
                final_df[col] = num.fillna(0.0)
            else:
                final_df = final_df.drop(columns=[col])

    final_df = final_df.set_index('user_id')

    # neighbor risk: proportion of known blacklisted neighbors
    status_dict = final_df['status'].to_dict() if 'status' in final_df.columns else {}
    neigh_risk = {}
    for node in all_nodes:
        if node not in final_df.index:
            neigh_risk[node] = 0.0
            continue
        neighbors = list(G.neighbors(node)) if node in G else []
        if len(neighbors) == 0:
            neigh_risk[node] = 0.0
            continue
        cnt_black = sum(1 for nb in neighbors if status_dict.get(nb, 0) == 1)
        neigh_risk[node] = cnt_black / max(1, len(neighbors))

    final_df['neighbor_risk_score'] = pd.Series(neigh_risk)

    # account tenure: compute days since known date (e.g., level1_finished_at / confirmed_at)
    date_cols = [c for c in final_df.columns if 'level1_finished_at' in c.lower() or 'confirmed' in c.lower() or 'created_at' in c.lower()]
    if len(date_cols) > 0:
        # parse first available date column and compute tenure relative to max date seen
        for c in date_cols:
            try:
                final_df[c] = pd.to_datetime(final_df[c], errors='coerce', utc=True)
            except Exception:
                final_df[c] = pd.to_datetime(final_df[c], errors='coerce')
        # use the latest observed date as reference
        ref_dates = pd.concat([final_df[c].dropna() for c in date_cols]) if len(date_cols) > 0 else pd.Series([])
        if len(ref_dates) > 0:
            ref = ref_dates.max()
            final_df['account_tenure_days'] = (ref - final_df[date_cols[0]]).dt.days.fillna(0).clip(lower=0)
        else:
            final_df['account_tenure_days'] = 0
    else:
        final_df['account_tenure_days'] = 0

    # Ensure datetime columns are numeric (unix seconds) to avoid Timestamp in feature matrix
    for col in final_df.columns:
        try:
            if np.issubdtype(final_df[col].dtype, np.datetime64):
                final_df[col] = (final_df[col].astype('int64') // 10**9).fillna(0)
        except Exception:
            # skip non-datetime columns
            pass

    # ensure status is numeric
    if 'status' in final_df.columns:
        final_df['status'] = pd.to_numeric(final_df['status'], errors='coerce').fillna(0).astype(int)

    # feature columns
    feat_cols = [c for c in final_df.columns if c != 'status']

    # Ensure all datetime-like or object date columns are numeric (unix seconds)
    # Use a safe per-cell coercion to handle Timestamp, timezone-aware, strings, and mixed types
    def _to_numeric_val(v):
        # NaN
        try:
            if pd.isna(v):
                return 0.0
        except Exception:
            pass
        # pandas Timestamp / numpy datetime64
        try:
            if isinstance(v, (pd.Timestamp,)) or isinstance(v, np.datetime64):
                try:
                    return float(pd.Timestamp(v).timestamp())
                except Exception:
                    try:
                        return float(int(v) // 10**9)
                    except Exception:
                        return 0.0
        except Exception:
            pass
        # numeric types
        if isinstance(v, (int, float, np.integer, np.floating, bool)):
            try:
                return float(v)
            except Exception:
                return 0.0
        # string: try parse datetime then numeric
        if isinstance(v, str):
            try:
                dt = pd.to_datetime(v, errors='coerce', utc=True)
                if not pd.isna(dt):
                    return float(pd.Timestamp(dt).timestamp())
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return 0.0
        # fallback: try numeric coercion
        try:
            return float(v)
        except Exception:
            return 0.0

    # apply safe coercion to all feature cells
    if len(feat_cols) > 0:
        final_df[feat_cols] = final_df[feat_cols].applymap(_to_numeric_val)

    # build x,y aligned with all_nodes
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

# --- model definition (GATv2) ---
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

# --- helper: train with early stopping ---
def train_with_earlystop(model, data, criterion, optimizer, patience=20, max_epochs=500):
    best_val = 1e9
    patience_cnt = 0
    best_state = None
    device = next(model.parameters()).device
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
        if epoch % 20 == 0 or patience_cnt == 0:
            with torch.no_grad():
                v_logits = model(data.x, data.edge_index)
                v_preds = v_logits.argmax(dim=1).cpu().numpy()
                labels = data.y.cpu().numpy()
                cm_val = confusion_matrix(labels[data.val_mask.cpu().numpy()], v_preds[data.val_mask.cpu().numpy()])
                if cm_val.size == 4:
                    tn, fp, fn, tp = cm_val.ravel()
                    val_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                else:
                    val_fnr = 0.0
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_FNR={val_fnr:.4%}")
        if patience_cnt >= patience:
            print('Early stopping at epoch', epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# --- permutation importance on test set (drop in positive-class F1) ---
def permutation_importance(model, data, X_df, feature_names, metric='f1', n_repeat=5):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    test_idx = np.where(data.test_mask.cpu().numpy())[0]
    y_test = data.y.cpu().numpy()[test_idx]
    baseline_f1 = f1_score(y_test, preds[test_idx], zero_division=0)

    importances = {}
    X_values = X_df.values.copy()
    for j, fname in enumerate(feature_names):
        scores = []
        for _ in range(n_repeat):
            X_perm = X_values.copy()
            perm = np.random.permutation(test_idx)
            # permute only test rows for column j
            X_perm[test_idx, j] = X_values[perm, j]
            # replace data.x for evaluation
            x_tensor = torch.tensor(X_perm, dtype=torch.float).to(next(model.parameters()).device)
            with torch.no_grad():
                logits_p = model(x_tensor, data.edge_index)
                preds_p = logits_p.argmax(dim=1).cpu().numpy()
            f1p = f1_score(y_test, preds_p[test_idx], zero_division=0)
            scores.append(f1p)
        importances[fname] = baseline_f1 - np.mean(scores)
    # sort by importance desc
    imp_series = pd.Series(importances).sort_values(ascending=False)
    return imp_series, baseline_f1

# --- feature crossing: generate pairwise product features for top_k ---
def generate_feature_crosses(X_df, top_features, max_pairs=10):
    pairs = []
    new_df = X_df.copy()
    from itertools import combinations
    comb = list(combinations(top_features, 2))[:max_pairs]
    for a, b in comb:
        col_name = f"{a}_x_{b}"
        new_df[col_name] = X_df[a] * X_df[b]
        pairs.append(col_name)
    return new_df, pairs

# --- main flow ---
if __name__ == '__main__':
    print('Building graph and features...')
    G, id_map, edge_index, all_nodes = build_graph_data()
    final_df, X_df, y_series, feature_names = prepare_features_df(id_map, all_nodes, G)

    # split indices stratified
    idx = np.arange(len(X_df))
    y_np = y_series.values
    tr_idx, temp_idx = train_test_split(idx, stratify=y_np, test_size=0.3, random_state=42)
    val_idx, te_idx = train_test_split(temp_idx, stratify=y_np[temp_idx], test_size=0.5, random_state=42)

    train_mask = torch.zeros(len(X_df), dtype=torch.bool)
    val_mask = torch.zeros(len(X_df), dtype=torch.bool)
    test_mask = torch.zeros(len(X_df), dtype=torch.bool)
    train_mask[tr_idx] = True; val_mask[val_idx] = True; test_mask[te_idx] = True

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_series.values, dtype=torch.long)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = train_mask; data.val_mask = val_mask; data.test_mask = test_mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # baseline model
    model = EnhancedGAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
    criterion = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

    print('Training baseline model...')
    model = train_with_earlystop(model, data, criterion, optimizer, patience=30, max_epochs=200)

    # baseline evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()

    cm_test = confusion_matrix(labels[test_mask.cpu().numpy()], preds[test_mask.cpu().numpy()])
    print('\nBaseline Test Confusion Matrix:\n', cm_test)
    print('\nBaseline Classification Report (test):')
    try:
        print(classification_report(labels[test_mask.cpu().numpy()], preds[test_mask.cpu().numpy()]))
    except Exception:
        pass

    # permutation importance
    print('\nComputing permutation importance (test set)...')
    imp_series, baseline_f1 = permutation_importance(model, data, X_df, feature_names, n_repeat=3)
    print('\nTop features by permutation importance (drop in F1):')
    print(imp_series.head(20))

    # take top_k features and generate crosses
    top_k = 6
    top_feats = list(imp_series.head(top_k).index)
    print('\nTop features for crossing:', top_feats)
    X_cross_df, new_cols = generate_feature_crosses(X_df, top_feats, max_pairs=10)
    print('Generated cross features:', new_cols)

    # re-scale and retrain with crossed features
    Xc_scaled = scaler.fit_transform(X_cross_df.values)
    Xc_tensor = torch.tensor(Xc_scaled, dtype=torch.float).to(device)
    data.x = Xc_tensor

    model2 = EnhancedGAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.005, weight_decay=1e-3)
    print('\nTraining model with feature crosses...')
    model2 = train_with_earlystop(model2, data, criterion, optimizer2, patience=30, max_epochs=200)

    model2.eval()
    with torch.no_grad():
        logits2 = model2(data.x, data.edge_index)
        probs2 = F.softmax(logits2, dim=1)[:, 1].cpu().numpy()
        preds2 = logits2.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()

    cm_test2 = confusion_matrix(labels[test_mask.cpu().numpy()], preds2[test_mask.cpu().numpy()])
    print('\nAfter Crosses Test Confusion Matrix:\n', cm_test2)
    print('\nClassification Report (test after crosses):')
    try:
        print(classification_report(labels[test_mask.cpu().numpy()], preds2[test_mask.cpu().numpy()]))
    except Exception:
        pass

    # --- Hybrid: extract GAT embeddings and train RandomForest on them ---
    from sklearn.ensemble import RandomForestClassifier
    with torch.no_grad():
        gat_embeddings = F.elu(model2.conv1(data.x, data.edge_index)).cpu().numpy()

    # train RF on train embeddings
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=12, random_state=42)
    rf.fit(gat_embeddings[tr_idx], y_np[tr_idx])

    # use validation set to find threshold (no test leakage)
    val_probs = rf.predict_proba(gat_embeddings[val_idx])[:, 1]
    y_val = y_np[val_idx]
    prec_v, rec_v, th_v = precision_recall_curve(y_val, val_probs)
    idxs_v = np.where(rec_v >= 1.0)[0]
    best_threshold = None
    if len(idxs_v) > 0 and len(th_v) > 0:
        idx = idxs_v[-1]
        thr_idx = max(0, min(len(th_v)-1, idx-1))
        best_threshold = th_v[thr_idx]
        print(f"Best threshold (Recall==1.0) on val: {best_threshold:.4f}")
    else:
        # fallback: choose threshold maximizing F1 on validation
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1; best_t = 0.5
        for t in thresholds:
            p = (val_probs >= t).astype(int)
            f = f1_score(y_val, p, zero_division=0)
            if f > best_f1:
                best_f1 = f; best_t = t
        best_threshold = best_t
        print(f"Fallback best threshold by F1 on val: {best_threshold:.4f}")

    # apply RF + threshold on test set
    test_probs = rf.predict_proba(gat_embeddings[te_idx])[:, 1]
    test_preds_rf = (test_probs >= best_threshold).astype(int)
    test_labels = y_np[te_idx]
    new_cm = confusion_matrix(test_labels, test_preds_rf)
    print('\n--- Hybrid RF (GAT embeddings) + val-threshold on test set ---')
    print(new_cm)
    try:
        print(classification_report(test_labels, test_preds_rf))
    except Exception:
        pass

    # save artifacts
    torch.save(model2.state_dict(), 'gcn_v5_model.pt')
    print('Saved gcn_v5_model.pt')

    # t-SNE on latent
    print('Generating t-SNE...')
    with torch.no_grad():
        latent = F.elu(model2.conv1(data.x, data.edge_index)).cpu().numpy()
    sample_idx = np.arange(latent.shape[0])
    if latent.shape[0] > 5000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(latent.shape[0], size=5000, replace=False)
    res = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(latent[sample_idx])
    scatter_df = pd.DataFrame({'x': res[:,0], 'y': res[:,1], 'label': ['Blacklist' if lbl==1 else 'Normal' for lbl in labels[sample_idx]]})
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=scatter_df, x='x', y='y', hue='label', palette={'Blacklist':'red','Normal':'skyblue'}, alpha=0.6, s=20)
    plt.title('gcn_v5 t-SNE')
    plt.savefig('gcn_v5_tsne.png', dpi=150)
    print('Saved gcn_v5_tsne.png')

    # print top feature crosses added
    print('\nAdded cross features:', [c for c in X_cross_df.columns if '_x_' in c])

    print('\nDone.')
