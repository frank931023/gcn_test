import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import copy

# --- Focal Loss implementation ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # pt = prob of true class
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                alpha = self.alpha.to(logits.device)
            else:
                alpha = self.alpha
            at = alpha[targets]
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- 1. 建圖與邊權重（保留但 GATv2 不使用 edge_weight） ---
def build_graph_data():
    shared_ip = pd.read_csv('../data/edge/shared_ip_edges.csv')
    crypto = pd.read_csv('../data/crypto_transfer.csv')

    G = nx.Graph()
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

    for (u, v), s in edge_stats.items():
        G.add_edge(u, v, count=s['count'], amount=s['amount'])

    all_nodes = sorted(list(G.nodes()))
    id_map = {n: i for i, n in enumerate(all_nodes)}

    # create edge_index (undirected)
    edge_list = []
    for u, v in G.edges():
        edge_list.append([id_map[u], id_map[v]])
        edge_list.append([id_map[v], id_map[u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return G, id_map, edge_index, all_nodes

# --- 2. 特徵整合（含 log1p 與 鄰居風險特徵） ---
def prepare_features(id_map, all_nodes, G):
    behavior_df = pd.read_csv('extended_features_analysis.csv')
    n2v_df = pd.read_csv('user_n2v.embeddings', sep=' ', skiprows=1, header=None)
    n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]

    final_df = behavior_df.merge(n2v_df, on='user_id', how='inner')

    # log1p on amount-like columns
    amount_cols = [c for c in final_df.columns if 'amount' in c.lower() or 'total' in c.lower()]
    if len(amount_cols) > 0:
        final_df[amount_cols] = final_df[amount_cols].apply(lambda s: np.log1p(pd.to_numeric(s, errors='coerce').fillna(0.0)))

    # convert date/object columns where possible, drop others
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

    # add neighbor-risk feature: proportion of labeled blacklist neighbors
    neigh_risk = {}
    for node in all_nodes:
        if node not in final_df.index:
            neigh_risk[node] = 0.0
            continue
        neighbors = list(G.neighbors(node)) if node in G else []
        if len(neighbors) == 0:
            neigh_risk[node] = 0.0
            continue
        cnt_black = 0
        for nb in neighbors:
            if nb in final_df.index:
                try:
                    if int(final_df.loc[nb].get('status', 0)) == 1:
                        cnt_black += 1
                except Exception:
                    pass
        neigh_risk[node] = cnt_black / max(1, len(neighbors))

    # ensure feature order
    feat_cols = [c for c in final_df.columns if c != 'status']
    feature_dim = len(feat_cols) + 1  # +1 for neigh_risk
    x_matrix = np.zeros((len(all_nodes), feature_dim), dtype=float)
    y_vector = np.zeros(len(all_nodes), dtype=int)

    for i, node in enumerate(all_nodes):
        if node in final_df.index:
            row = final_df.loc[node]
            vals = row[feat_cols].values.astype(float)
            x_matrix[i, :len(vals)] = vals
            x_matrix[i, -1] = neigh_risk.get(node, 0.0)
            y_vector[i] = int(row.get('status', 0))
        else:
            # missing node -> zeros
            x_matrix[i, :] = 0.0
            y_vector[i] = 0

    return torch.tensor(x_matrix, dtype=torch.float), torch.tensor(y_vector, dtype=torch.long)

# --- 3. 主程式: 建圖、特徵、Data、訓練 ---
G, id_map, edge_index, all_nodes = build_graph_data()
x, y = prepare_features(id_map, all_nodes, G)

data = Data(x=x, edge_index=edge_index, y=y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class N2V_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=1, dropout=0.3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.elu(h1)
        h2 = self.conv2(h1, edge_index)
        h2 = F.elu(h2)
        combined = torch.cat([h2, x], dim=1)
        return self.classifier(combined)

model = N2V_GAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)

# --- loss: FocalLoss ---
# alpha 可依類別不平衡調整，例如給黑名單較高權重但不致過度偏斜
alpha = [0.25, 0.75]
criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

# scaling (注意 log1p 已經處理 amount-like 欄位)
scaler = StandardScaler()
X_np = data.x.numpy() if isinstance(data.x, torch.Tensor) else np.array(data.x)
X_scaled = scaler.fit_transform(X_np)
data.x = torch.tensor(X_scaled, dtype=torch.float)

# train/val/test split (stratified by label)
y_np = data.y.numpy()
idx = np.arange(len(y_np))
tr_idx, temp_idx, y_tr, y_temp = train_test_split(idx, y_np, stratify=y_np, test_size=0.30, random_state=42)
val_idx, te_idx, y_val, y_te = train_test_split(temp_idx, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

train_mask = torch.zeros(len(y_np), dtype=torch.bool)
val_mask = torch.zeros(len(y_np), dtype=torch.bool)
test_mask = torch.zeros(len(y_np), dtype=torch.bool)
train_mask[tr_idx] = True
val_mask[val_idx] = True
test_mask[te_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

data = data.to(device)

# training with early stopping
best_val = 1e9
patience = 30
patience_cnt = 0
best_state = None

for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        v_logits = model(data.x, data.edge_index)
        val_loss = criterion(v_logits[data.val_mask], data.y[data.val_mask]).item()
        v_preds = v_logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        cm_val = confusion_matrix(labels[val_mask.cpu().numpy()], v_preds[val_mask.cpu().numpy()])
        if cm_val.size == 4:
            tn, fp, fn, tp = cm_val.ravel()
            val_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            val_fnr = 0.0

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_cnt = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        patience_cnt += 1

    if epoch % 20 == 0 or patience_cnt == 0:
        print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_FNR={val_fnr:.4%}")

    if patience_cnt >= patience:
        print('Early stopping at epoch', epoch)
        break

# restore best
if best_state is not None:
    model.load_state_dict(best_state)

# final evaluation on train/val/test masks
model.eval()
with torch.no_grad():
    final_logits = model(data.x, data.edge_index)
    final_preds = final_logits.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()

cm_train = confusion_matrix(labels[train_mask.cpu().numpy()], final_preds[train_mask.cpu().numpy()])
cm_val = confusion_matrix(labels[val_mask.cpu().numpy()], final_preds[val_mask.cpu().numpy()])
cm_test = confusion_matrix(labels[test_mask.cpu().numpy()], final_preds[test_mask.cpu().numpy()])

print('\nTrain Confusion Matrix:\n', cm_train)
print('\nVal Confusion Matrix:\n', cm_val)
print('\nTest Confusion Matrix:\n', cm_test)

print('\nClassification report (test):')
try:
    print(classification_report(labels[test_mask.cpu().numpy()], final_preds[test_mask.cpu().numpy()]))
except Exception:
    pass

# precision-recall and best threshold for FNR=0 on test set
with torch.no_grad():
    probs = F.softmax(final_logits, dim=1)[:, 1].cpu().numpy()
    precisions, recalls, thresholds = precision_recall_curve(labels[test_mask.cpu().numpy()], probs[test_mask.cpu().numpy()])
    idxs = np.where(recalls >= 1.0)[0]
    best_threshold = None
    if len(idxs) > 0 and len(thresholds) > 0:
        idx = idxs[-1]
        thr_idx = max(0, min(len(thresholds)-1, idx-1))
        best_threshold = thresholds[thr_idx]
        print(f"達成 FNR=0 的最佳判斷門檻 (test set): {best_threshold:.4f}")
    else:
        print('找不到在 test 上使 Recall==1.0 的閾值')

# t-SNE visualization on latent features (first attention layer)
print('\n產生 t-SNE 視覺化...')
with torch.no_grad():
    latent = F.elu(model.conv1(data.x, data.edge_index)).cpu().numpy()

num_nodes = latent.shape[0]
if num_nodes > 5000:
    rng = np.random.RandomState(42)
    idxs_sample = rng.choice(num_nodes, size=5000, replace=False)
    latent_sample = latent[idxs_sample]
    labels_sample = labels[idxs_sample]
else:
    latent_sample = latent
    labels_sample = labels

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
res = tsne.fit_transform(latent_sample)

scatter_df = pd.DataFrame({'x': res[:,0], 'y': res[:,1], 'label': ['Blacklist' if l==1 else 'Normal' for l in labels_sample]})
plt.figure(figsize=(10,8))
sns.scatterplot(data=scatter_df, x='x', y='y', hue='label', palette={'Blacklist':'red','Normal':'skyblue'}, alpha=0.6, s=20)
plt.title('GATv2 latent t-SNE')
plt.savefig('gcn_v4_tsne.png', dpi=150)
print('t-SNE saved to gcn_v4_tsne.png')

# save model
torch.save(model.state_dict(), 'gcn_v4_model.pt')
print('Model saved to gcn_v4_model.pt')
