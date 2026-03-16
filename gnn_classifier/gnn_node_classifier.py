"""
GCN / GraphSAGE 用戶黑名單分類
- 節點 = 用戶 (只用 train_label.csv 中有標籤的用戶)
- 邊   = crypto_transfer.csv 中的轉帳關係 (user_id → relation_user_id，只保留 sub_kind==1 的內部轉帳)
- 節點特徵 = [out_degree, in_degree, asym, total, log1p_out, log1p_in]
- 目標 = status (0=正常, 1=黑名單)
- 同時跑 GCN 和 GraphSAGE，最後比較結果
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)

# ── 路徑設定 ──────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
EPOCHS       = 200
LR           = 0.005
HIDDEN       = 64
DROPOUT      = 0.5

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# ── 1. 讀取資料 ───────────────────────────────────────────
print("\n[1] 讀取資料...")
crypto = pd.read_csv(os.path.join(DATA_DIR, 'crypto_transfer.csv'))
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_label.csv'))

print(f"  crypto_transfer: {len(crypto):,} 筆")
print(f"  train_label    : {len(labels):,} 筆  (黑名單={labels['status'].sum():,})")

# ── 2. 只保留有標籤的用戶，建立 user→node_id 映射 ─────────
labeled_users = set(labels['user_id'].astype(str))
user_to_nid   = {u: i for i, u in enumerate(sorted(labeled_users))}
num_nodes     = len(user_to_nid)
print(f"\n[2] 有標籤用戶數: {num_nodes:,}")

# ── 3. 建立邊 (sub_kind==1 內部轉帳，relation_user_id 轉整數字串) ──
print("\n[3] 建立圖邊...")
internal = crypto[crypto['sub_kind'] == 1].copy()
# relation_user_id 是 float (e.g. 83439.0)，需先轉 int 再轉 str
internal = internal[internal['relation_user_id'].notna()].copy()
internal['src'] = internal['user_id'].astype(int).astype(str)
internal['dst'] = internal['relation_user_id'].astype(int).astype(str)

# 只保留兩端都在有標籤用戶中的邊
mask = internal['src'].isin(labeled_users) & internal['dst'].isin(labeled_users)
edges = internal[mask][['src', 'dst']].drop_duplicates()
print(f"  內部轉帳總筆數: {len(internal):,}")
print(f"  兩端均有標籤的邊: {len(edges):,}")

src_ids = edges['src'].map(user_to_nid).values
dst_ids = edges['dst'].map(user_to_nid).values

# 無向圖：加入反向邊
edge_index = torch.tensor(
    [np.concatenate([src_ids, dst_ids]),
     np.concatenate([dst_ids, src_ids])],
    dtype=torch.long
)

# ── 4. 計算節點特徵 ───────────────────────────────────────
print("\n[4] 計算節點特徵...")

# 用全部 internal 轉帳計算 degree（不限兩端都有標籤）
# internal 已在上方做過 int->str 轉換
out_deg_all = internal.groupby('src')['dst'].count().rename('out_deg')
in_deg_all  = internal.groupby('dst')['src'].count().rename('in_deg')

feat_df = pd.DataFrame({'user_id': sorted(labeled_users)})
feat_df = feat_df.merge(out_deg_all.reset_index().rename(columns={'src':'user_id'}), on='user_id', how='left')
feat_df = feat_df.merge(in_deg_all.reset_index().rename(columns={'dst':'user_id'}),  on='user_id', how='left')
feat_df = feat_df.fillna(0)

feat_df['total']      = feat_df['out_deg'] + feat_df['in_deg']
feat_df['asym']       = np.where(
    feat_df['total'] > 0,
    (feat_df['out_deg'] - feat_df['in_deg']) / feat_df['total'],
    0.0
)
feat_df['log1p_out']  = np.log1p(feat_df['out_deg'])
feat_df['log1p_in']   = np.log1p(feat_df['in_deg'])

feature_cols = ['out_deg', 'in_deg', 'asym', 'total', 'log1p_out', 'log1p_in']
X = feat_df[feature_cols].values.astype(np.float32)

# 標準化
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X      = (X - X_mean) / X_std

node_features = torch.tensor(X, dtype=torch.float)
print(f"  節點特徵矩陣: {node_features.shape}")

# ── 5. 標籤 ──────────────────────────────────────────────
label_map = dict(zip(labels['user_id'].astype(str), labels['status']))
y_list    = [label_map[u] for u in sorted(labeled_users)]
y         = torch.tensor(y_list, dtype=torch.long)
print(f"  標籤分布: 正常={( y==0).sum().item():,}  黑名單={(y==1).sum().item():,}")

# ── 6. Train/Test split (stratified) ─────────────────────
print("\n[5] 切分訓練/測試集...")
all_idx = np.arange(num_nodes)
train_idx, test_idx = train_test_split(
    all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y_list
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True

print(f"  訓練: {train_mask.sum().item():,}  測試: {test_mask.sum().item():,}")

# ── 7. 建立 PyG Data ──────────────────────────────────────
data = Data(
    x          = node_features,
    edge_index = edge_index,
    y          = y,
    num_nodes  = num_nodes
).to(device)

train_mask = train_mask.to(device)
test_mask  = test_mask.to(device)

# 計算 class weight 處理不平衡
n_black = (y == 1).sum().item()
n_white = (y == 0).sum().item()
pos_weight = n_white / n_black
print(f"\n  正負樣本比: 1:{pos_weight:.1f}  → class_weight 黑名單 x{pos_weight:.1f}")

# ── 8. 模型定義 ───────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, out_ch)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.lin(x)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, out_ch)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.lin(x)


# ── 9. 訓練函數 ───────────────────────────────────────────
def train_model(model_name, model):
    print(f"\n{'='*55}")
    print(f"  訓練 {model_name}")
    print(f"{'='*55}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    # weighted cross-entropy
    weight = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)

    best_f1   = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=weight)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                preds  = logits[train_mask].argmax(dim=1).cpu().numpy()
                f1     = f1_score(data.y[train_mask].cpu().numpy(), preds, zero_division=0)
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train F1(黑): {f1:.4f}")

            if f1 > best_f1:
                best_f1    = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 載入最佳權重
    if best_state:
        model.load_state_dict(best_state)

    return model


# ── 10. 評估函數 ──────────────────────────────────────────
def evaluate(model_name, model):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        true   = data.y.cpu().numpy()

    for split_name, mask in [('Train', train_mask.cpu()), ('Test', test_mask.cpu())]:
        m_true  = true[mask]
        m_preds = preds[mask]
        m_probs = probs[mask]

        print(f"\n── {model_name} │ {split_name} ──────────────────────────")
        print(classification_report(m_true, m_preds,
                                    target_names=['正常(0)', '黑名單(1)'],
                                    digits=4, zero_division=0))
        cm = confusion_matrix(m_true, m_preds)
        print(f"  混淆矩陣:\n{cm}")
        try:
            auc_roc = roc_auc_score(m_true, m_probs)
            auc_pr  = average_precision_score(m_true, m_probs)
            print(f"  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}")
        except Exception:
            pass

    # 儲存測試集預測
    test_true  = true[test_mask.cpu()]
    test_preds = preds[test_mask.cpu()]
    test_probs = probs[test_mask.cpu()]
    out_df = pd.DataFrame({
        'true': test_true,
        'pred': test_preds,
        'prob_black': test_probs
    })
    fname = os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ","_")}_test_preds.csv')
    out_df.to_csv(fname, index=False)
    print(f"\n  測試集預測已儲存: {fname}")


# ── 11. 執行 ─────────────────────────────────────────────
in_ch = node_features.shape[1]

gcn_model   = train_model("GCN",       GCN(in_ch, HIDDEN, 2, DROPOUT))
sage_model  = train_model("GraphSAGE", GraphSAGE(in_ch, HIDDEN, 2, DROPOUT))

print("\n\n" + "="*55)
print("  最終評估結果")
print("="*55)
evaluate("GCN",       gcn_model)
evaluate("GraphSAGE", sage_model)

# 儲存模型
torch.save(gcn_model.state_dict(),  os.path.join(OUTPUT_DIR, 'gcn_model.pt'))
torch.save(sage_model.state_dict(), os.path.join(OUTPUT_DIR, 'graphsage_model.pt'))
print("\n✓ 模型已儲存至 gnn_classifier/outputs/")
