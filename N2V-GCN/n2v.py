import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# 1. 構建全域圖 (包含 IP 關聯與轉帳關係)
shared_ip = pd.read_csv('../data/edge/shared_ip_edges.csv')
crypto = pd.read_csv('../data/crypto_transfer.csv')

G = nx.Graph()

# 加入 IP 共享邊
for _, row in shared_ip.iterrows():
    G.add_edge(row['user_id_1'], row['user_id_2'], weight=1)

# 加入內轉關係邊 (如果存在關係)
crypto_internal = crypto[crypto['relation_user_id'].notnull()]
for _, row in crypto_internal.iterrows():
    G.add_edge(int(row['user_id']), int(row['relation_user_id']), weight=2)

# 2. 執行 Node2Vec 產生結構向量
# p=1, q=0.5 代表更傾向於探索（發現洗錢鏈結構）
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, p=1, q=0.5, workers=4)
model_n2v = node2vec.fit(window=10, min_count=1)

# 儲存向量方便後續讀取
model_n2v.wv.save_word2vec_format("user_n2v.embeddings")