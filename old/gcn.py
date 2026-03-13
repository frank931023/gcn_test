import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec, GCNConv
from torch.nn import Linear
from tqdm import tqdm

# ==========================================
# 步驟 1: 讀取你的真實 Table
# ==========================================
# 請確保檔名與你的實際路徑一致
user_info_df = pd.read_csv('data/user_info.csv')
crypto_transfer_df = pd.read_csv('data/crypto_transfer.csv')
train_label_df = pd.read_csv('data/train_label.csv')

# ==========================================
# 步驟 2: 萃取「黑名單用戶」集合 (Set)
# ==========================================
# 根據你的 user_info 表：status 0 是正常，1 是黑名單
bad_users = set(train_label_df[train_label_df['status'] == 1]['user_id'].values.astype(str))  # 確保都是字串格式
print(f"資料庫中共有 {len(bad_users)} 名黑名單用戶。")

# ==========================================
# 步驟 3: 標準化 crypto_transfer 並貼上交易標籤
# ==========================================
# 我們要建立一個乾淨的 list 來存放整理好的資料
clean_records = []

for index, row in tqdm(crypto_transfer_df.iterrows(), total=len(crypto_transfer_df), desc="Processing transactions"):
    src = None
    dst = None
    
    # 金額轉換 (根據你的 PDF 說明，真實數量需乘上 1e-8)
    amount = float(row['ori_samount']) * 1e-8 
    
    # --- 判斷交易方向邏輯 ---
    if row['sub_kind'] == 1:
        # 【內部轉帳】
        src = row['user_id']
        dst = row['relation_user_id']
        
    elif row['sub_kind'] == 0:
        # 【外部提充】
        if row['kind'] == 0:
            # 入金 (Deposit): 外部錢包 -> 交易所用戶
            src = row['from_wallet_hash']
            dst = row['user_id']
        elif row['kind'] == 1:
            # 出金 (Withdrawal): 交易所用戶 -> 外部錢包
            src = row['user_id']
            dst = row['to_wallet_hash']

    # --- 確保起點與終點都有抓到，且不是空值 ---
    if pd.notna(src) and pd.notna(dst):

        src = str(src)  # 確保都是字串格式
        dst = str(dst)
        
        # 【核心標籤邏輯】：只要 src 或 dst 其中一方是黑名單，這筆交易就是洗錢交易 (1)
        is_laundering = 1 if (src in bad_users or dst in bad_users) else 0
        
        # 將整理好的資料存入
        clean_records.append({
            'src_account': src,
            'dst_account': dst,
            'ori_samount': amount,
            'is_laundering': is_laundering
        })

# ==========================================
# 步驟 4: 轉換為我們剛才 N2V-GCN 程式碼需要的 DataFrame 格式
# ==========================================
standardized_df = pd.DataFrame(clean_records)

print("\n--- 整理完成的標準化交易表 ---")
print(standardized_df.head())

print("\n--- 交易標籤分佈統計 ---")
print(standardized_df['is_laundering'].value_counts())

all_accounts = pd.concat([
    standardized_df['src_account'],
    standardized_df['dst_account']
]).unique()

account_to_id = {acc: i for i, acc in enumerate(all_accounts)}

standardized_df['src_id'] = standardized_df['src_account'].map(account_to_id)
standardized_df['dst_id'] = standardized_df['dst_account'].map(account_to_id)

num_nodes = len(account_to_id)

print(f"\n總共有 {num_nodes} 個獨特的帳戶，已經被編碼成整數 ID。")


# 2. 切分訓練集與測試集 (依論文：測試集保持原始不平衡比例)
train_df, test_df = train_test_split(
    standardized_df, 
    test_size=0.2, 
    stratify=standardized_df['is_laundering'], 
    random_state=42
)

print(f"原始訓練集洗錢比例: {train_df['is_laundering'].mean():.4f}")

# 3. 論文的重採樣 (Resampling)
X_train = train_df[['src_account', 'dst_account', 'ori_samount']]
y_train = train_df['is_laundering']

# 使用 SMOTENC: 告訴演算法第 0 和第 1 欄 (src, dst) 是類別，不能拿來算小數點平均！
# sampling_strategy 可以自己調，論文是把少數類別拉高，多數類別降低
smote_nc = SMOTENC(categorical_features=[0, 1], random_state=42, sampling_strategy=0.1)
near_miss = NearMiss(version=1, sampling_strategy=1.0)

# Pipeline: 先 SMOTE 再 NearMiss
resample_pipe = Pipeline([('smotenc', smote_nc), ('nearmiss', near_miss)])
X_res, y_res = resample_pipe.fit_resample(X_train, y_train)

train_resampled_df = pd.DataFrame(X_res, columns=['src_account', 'dst_account', 'ori_samount'])
train_resampled_df['is_laundering'] = y_res

print(f"重採樣後訓練集洗錢比例: {train_resampled_df['is_laundering'].mean():.4f}")

train_edge_index = torch.tensor([
    train_resampled_df['src_id'].values,
    train_resampled_df['dst_id'].values
], dtype=torch.long)

print(f"訓練圖節點數: {num_nodes}")
print(f"訓練圖邊數: {train_edge_index.shape[1]}")

# ==========================================
# 階段 4: Node2Vec 完整訓練迴圈
# ==========================================
print("\n初始化 Node2Vec 模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用的運算裝置: {device}")

n2v_model = Node2Vec(
    train_edge_index, 
    embedding_dim=128,      # 輸出的向量維度
    walk_length=20,         # 每次漫步走幾步
    context_size=10,        # 視窗大小 (skip-gram)
    walks_per_node=10,      # 每個節點發起幾次漫步
    num_negative_samples=1, # 負採樣比例
    p=1.0,                  # Return parameter (論文設定)
    q=0.5,                  # In-out parameter (論文設定)
    sparse=True             # 必須設為 True 才能處理大量節點
).to(device)

# 1. 建立 Node2Vec 專屬的 DataLoader 
# 注意: 在 Windows 系統上，num_workers 建議設為 0 以避免多進程錯誤
loader = n2v_model.loader(batch_size=128, shuffle=True, num_workers=0)

# 2. 由於 sparse=True，必須使用 PyTorch 的 SparseAdam 優化器
optimizer = torch.optim.SparseAdam(list(n2v_model.parameters()), lr=0.01)

# 3. 定義單個 Epoch 的訓練函數
def train_node2vec():
    n2v_model.train()
    total_loss = 0
    
    # 每次迭代會自動產出正樣本 (pos_rw) 與負樣本 (neg_rw) 的隨機漫步路徑
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        
        # 將路徑放入 GPU/CPU，並計算 Loss
        loss = n2v_model.loss(pos_rw.to(device), neg_rw.to(device))
        
        # 反向傳播與權重更新
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# 4. 執行訓練迴圈 (設定 50 個 Epoch 作為範例)
epochs = 50
print(f"\n開始訓練 Node2Vec (共 {epochs} Epochs)...")

for epoch in tqdm(range(1, epochs + 1), desc="Training Node2Vec"):
    loss = train_node2vec()
    # 每 5 個 epoch 印出一次進度
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# ==========================================
# 階段 5: 提取最終向量
# ==========================================
n2v_model.eval()
with torch.no_grad():
    # 呼叫模型本身就會回傳所有節點的 Embedding Matrix
    node_embeddings = n2v_model() 

print("\n🎉 Node2Vec 訓練完成！")
print(f"最終萃取出的特徵矩陣維度 (節點數, 向量維度): {node_embeddings.shape}")

# 接下來，這個 `node_embeddings` 就可以無縫餵給 GCN 分類器了！
# 初始化 GCN 模型與優化器
gcn_hidden_dim = 64
gcn_model = N2V_GCN_EdgeClassifier(in_channels=128, hidden_channels=gcn_hidden_dim).to(device)

optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 準備 GCN 訓練資料
# 使用 .detach() 凍結 N2V 向量，避免反向傳播改變已經訓練好的 N2V 權重 (節省記憶體)
x_input = node_embeddings.detach() 
train_labels = torch.tensor(train_resampled_df['is_laundering'].values, dtype=torch.long).to(device)

print("\n開始訓練 GCN 分類器...")
gcn_epochs = 100

gcn_model.train()
for epoch in tqdm(range(1, gcn_epochs + 1), desc="Training GCN"):
    optimizer_gcn.zero_grad()
    
    # 預測訓練集中的交易
    out = gcn_model(
        x=x_input, 
        edge_index=train_edge_index,  # 圖的結構
        query_edges=train_edge_index  # 這次要求預測的邊 (訓練時就是訓練集的邊)
    )
    
    loss = criterion(out, train_labels)
    loss.backward()
    optimizer_gcn.step()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f'GCN Epoch: {epoch:03d}, Loss: {loss.item():.4f}')

# ==========================================
# 階段 7: 模型評估 (使用 Test Set)
# ==========================================
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("\n進行測試集評估...")

# 1. 準備測試集的 Query Edges
test_df['src_id'] = test_df['src_account'].map(account_to_id)
test_df['dst_id'] = test_df['dst_account'].map(account_to_id)

test_query_edges = torch.tensor([
    test_df['src_id'].values,
    test_df['dst_id'].values
], dtype=torch.long).to(device)

test_labels = test_df['is_laundering'].values # numpy array for sklearn

# 2. 進行預測
gcn_model.eval()
with torch.no_grad():
    # 注意：預測時，背景的圖結構 (edge_index) 依然使用訓練集的結構，以防資料洩漏
    test_out = gcn_model(
        x=x_input, 
        edge_index=train_edge_index, 
        query_edges=test_query_edges
    )
    
    # 轉成機率值 (為了算 AUC)
    test_probs = F.softmax(test_out, dim=1)[:, 1].cpu().numpy()
    # 取得最終預測類別 (0 或 1)
    test_preds = test_out.argmax(dim=1).cpu().numpy()

# 3. 印出評估報告
print("\n" + "="*40)
print("🏆 N2V-GCN 模型測試結果報告")
print("="*40)

# 計算 AUC (在極度不平衡的 AML 資料中最具參考價值)
auc_score = roc_auc_score(test_labels, test_probs)
print(f"ROC-AUC 預測能力分數: {auc_score:.4f} (越接近 1 越強)\n")

print("📊 混淆矩陣 (Confusion Matrix):")
print(confusion_matrix(test_labels, test_preds))

print("\n📑 詳細分類報告:")
print(classification_report(test_labels, test_preds, target_names=['正常交易 (0)', '洗錢交易 (1)']))

