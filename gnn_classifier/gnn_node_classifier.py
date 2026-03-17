"""
GCN / GraphSAGE 用戶黑名單分類
比較兩種邊的建構方式：
  - mode='internal' : 只用 sub_kind==1 內部轉帳（原版）
  - mode='all'      : sub_kind==0 鏈上交易 + sub_kind==1 內部轉帳（新版）

鏈上交易邊的建構邏輯：
  - kind=1 (出金) sub_kind=0：src=user_id, dst=to_wallet_hash
  - kind=0 (入金) sub_kind=0：src=from_wallet_hash, dst=user_id
  共享錢包橋接：若 user A 出金到 wallet W，user B 從 wallet W 入金，則 A-B 建邊
  這樣可以捕捉「A 出金到某錢包，B 從同一錢包入金」的洗錢模式
"""

import os
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
print(f"  sub_kind 分布: {crypto['sub_kind'].value_counts().to_dict()}")

# ── 2. 有標籤用戶集合 ─────────────────────────────────────
labeled_users = set(labels['user_id'].astype(str))
user_to_nid   = {u: i for i, u in enumerate(sorted(labeled_users))}
num_nodes     = len(user_to_nid)
print(f"\n[2] 有標籤用戶數: {num_nodes:,}")

# ── 3. 標籤 & Train/Test split ───────────────────────────
label_map = dict(zip(labels['user_id'].astype(str), labels['status']))
y_list    = [label_map[u] for u in sorted(labeled_users)]
y         = torch.tensor(y_list, dtype=torch.long)

all_idx = np.arange(num_nodes)
train_idx, test_idx = train_test_split(
    all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_list
)
train_mask_base = torch.zeros(num_nodes, dtype=torch.bool)
test_mask_base  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask_base[train_idx] = True
test_mask_base[test_idx]   = True

n_black    = (y == 1).sum().item()
n_white    = (y == 0).sum().item()
pos_weight = n_white / n_black

# ── 4. 邊建構函式 ─────────────────────────────────────────
def build_edges(mode: str):
    """
    mode='internal' : 只用 sub_kind==1
    mode='all'      : sub_kind==0 鏈上 + sub_kind==1 內部
    """
    edge_pairs = []

    # --- sub_kind==1 內部轉帳 ---
    internal = crypto[crypto['sub_kind'] == 1].copy()
    internal = internal[internal['relation_user_id'].notna()].copy()
    internal['src'] = internal['user_id'].astype(int).astype(str)
    internal['dst'] = internal['relation_user_id'].astype(int).astype(str)
    mask_int = internal['src'].isin(labeled_users) & internal['dst'].isin(labeled_users)
    ep_int = internal[mask_int][['src', 'dst']].drop_duplicates()
    edge_pairs.append(ep_int)
    print(f"  [internal] 兩端有標籤邊: {len(ep_int):,}")

    if mode == 'all':
        # --- sub_kind==0 鏈上交易：共享錢包橋接 ---
        # 邏輯：若 user A 出金到 wallet W，user B 從 wallet W 入金 → A-B 建邊
        onchain = crypto[crypto['sub_kind'] == 0].copy()
        onchain['user_id_str'] = onchain['user_id'].astype(int).astype(str)

        # 出金 kind=1：user → to_wallet_hash
        out_tx = onchain[onchain['kind'] == 1][['user_id_str', 'to_wallet_hash']].copy()
        out_tx = out_tx[out_tx['user_id_str'].isin(labeled_users)]
        out_tx = out_tx.rename(columns={'user_id_str': 'user', 'to_wallet_hash': 'wallet'})
        out_tx['wallet'] = out_tx['wallet'].astype(str)

        # 入金 kind=0：from_wallet_hash → user
        in_tx = onchain[onchain['kind'] == 0][['user_id_str', 'from_wallet_hash']].copy()
        in_tx = in_tx[in_tx['user_id_str'].isin(labeled_users)]
        in_tx = in_tx.rename(columns={'user_id_str': 'user', 'from_wallet_hash': 'wallet'})
        in_tx['wallet'] = in_tx['wallet'].astype(str)

        # 共享錢包橋接：出金 user × 入金 user（透過同一 wallet）
        bridge = out_tx.merge(in_tx, on='wallet', suffixes=('_out', '_in'))
        bridge = bridge[bridge['user_out'] != bridge['user_in']]  # 排除自己
        ep_bridge = bridge[['user_out', 'user_in']].rename(
            columns={'user_out': 'src', 'user_in': 'dst'}
        ).drop_duplicates()

        print(f"  [onchain bridge] 共享錢包橋接邊: {len(ep_bridge):,}")
        print(f"    (涉及出金 user: {out_tx['user'].nunique():,}, 入金 user: {in_tx['user'].nunique():,})")
        edge_pairs.append(ep_bridge)

    all_edges = pd.concat(edge_pairs, ignore_index=True).drop_duplicates()
    print(f"  [{mode}] 總邊數（去重）: {len(all_edges):,}")

    src_ids = all_edges['src'].map(user_to_nid).values
    dst_ids = all_edges['dst'].map(user_to_nid).values

    edge_index = torch.tensor(
        [np.concatenate([src_ids, dst_ids]),
         np.concatenate([dst_ids, src_ids])],
        dtype=torch.long
    )
    return edge_index, all_edges

# ── 5. 節點特徵建構函式 ───────────────────────────────────
def build_features(all_edges):
    out_deg = all_edges.groupby('src')['dst'].count().rename('out_deg')
    in_deg  = all_edges.groupby('dst')['src'].count().rename('in_deg')

    feat_df = pd.DataFrame({'user_id': sorted(labeled_users)})
    feat_df = feat_df.merge(out_deg.reset_index().rename(columns={'src': 'user_id'}),
                            on='user_id', how='left')
    feat_df = feat_df.merge(in_deg.reset_index().rename(columns={'dst': 'user_id'}),
                            on='user_id', how='left')
    feat_df = feat_df.fillna(0)
    feat_df['total']     = feat_df['out_deg'] + feat_df['in_deg']
    feat_df['asym']      = np.where(
        feat_df['total'] > 0,
        (feat_df['out_deg'] - feat_df['in_deg']) / feat_df['total'], 0.0)
    feat_df['log1p_out'] = np.log1p(feat_df['out_deg'])
    feat_df['log1p_in']  = np.log1p(feat_df['in_deg'])

    X = feat_df[['out_deg', 'in_deg', 'asym', 'total', 'log1p_out', 'log1p_in']].values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return torch.tensor(X, dtype=torch.float)

# ── 6. 模型定義 ───────────────────────────────────────────
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

# ── 7. 訓練 & 評估函式 ────────────────────────────────────
def train_model(model_name, model, data, train_mask):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    weight = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=weight)
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = model(data.x, data.edge_index)[train_mask].argmax(dim=1).cpu().numpy()
                f1    = f1_score(data.y[train_mask].cpu().numpy(), preds, zero_division=0)
            print(f"    Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1    = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model_name, model, data, train_mask, test_mask, mode):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        true   = data.y.cpu().numpy()

    results = {}
    for split_name, mask in [('Train', train_mask.cpu()), ('Test', test_mask.cpu())]:
        m_true, m_preds, m_probs = true[mask], preds[mask], probs[mask]
        print(f"\n  ── {model_name} [{mode}] │ {split_name} ──")
        print(classification_report(m_true, m_preds,
                                    target_names=['正常(0)', '黑名單(1)'],
                                    digits=4, zero_division=0))
        cm = confusion_matrix(m_true, m_preds)
        print(f"  混淆矩陣:\n{cm}")
        try:
            auc_roc = roc_auc_score(m_true, m_probs)
            auc_pr  = average_precision_score(m_true, m_probs)
            print(f"  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}")
            if split_name == 'Test':
                results = {
                    'model': model_name, 'mode': mode,
                    'auc_roc': auc_roc, 'auc_pr': auc_pr,
                    'f1': f1_score(m_true, m_preds, zero_division=0),
                }
        except Exception:
            pass

    # 儲存測試集預測
    test_df = pd.DataFrame({
        'true': true[test_mask.cpu()],
        'pred': preds[test_mask.cpu()],
        'prob_black': probs[test_mask.cpu()]
    })
    fname = os.path.join(OUTPUT_DIR, f'{model_name.lower()}_{mode}_test_preds.csv')
    test_df.to_csv(fname, index=False)
    return results

# ════════════════════════════════════════════════════════════════
# 主流程：跑兩種 mode 並比較
# ════════════════════════════════════════════════════════════════
all_results = []

for mode in ['internal', 'all']:
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*60}")

    print("\n[3] 建立圖邊...")
    edge_index, all_edges = build_edges(mode)

    print("\n[4] 計算節點特徵...")
    node_features = build_features(all_edges)
    print(f"  節點特徵矩陣: {node_features.shape}")
    print(f"  標籤分布: 正常={(y==0).sum().item():,}  黑名單={(y==1).sum().item():,}")
    print(f"  正負樣本比: 1:{pos_weight:.1f}")

    data = Data(
        x=node_features, edge_index=edge_index,
        y=y, num_nodes=num_nodes
    ).to(device)
    train_mask = train_mask_base.to(device)
    test_mask  = test_mask_base.to(device)

    in_ch = node_features.shape[1]

    for model_name, model_cls in [("GCN", GCN), ("GraphSAGE", GraphSAGE)]:
        print(f"\n  訓練 {model_name} [{mode}]...")
        model = train_model(model_name, model_cls(in_ch, HIDDEN, 2, DROPOUT),
                            data, train_mask)
        res = evaluate(model_name, model, data, train_mask, test_mask, mode)
        all_results.append(res)
        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR, f'{model_name.lower()}_{mode}.pt'))

# ── 比較表 ────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  最終比較：internal-only vs all (onchain + internal)")
print("="*60)
res_df = pd.DataFrame([r for r in all_results if r])
if not res_df.empty:
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'mode_comparison.csv'), index=False)
    print(f"\n比較表已儲存: {OUTPUT_DIR}/mode_comparison.csv")
