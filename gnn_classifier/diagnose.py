import pandas as pd

crypto = pd.read_csv('data/crypto_transfer.csv')
labels = pd.read_csv('data/train_label.csv')
labeled = set(labels['user_id'].astype(str))

print('=== sub_kind 分布 ===')
print(crypto['sub_kind'].value_counts())
print('\n=== kind 分布 ===')
print(crypto['kind'].value_counts())

crypto['src'] = crypto['user_id'].astype(str)
crypto['dst'] = crypto['relation_user_id'].astype(str)

rel_notnull = crypto['relation_user_id'].notna()
print(f'\nrelation_user_id 非空: {rel_notnull.sum()} / {len(crypto)}')
print(f'user_id 在 labeled 中: {crypto["src"].isin(labeled).sum()}')
print(f'relation_user_id 在 labeled 中: {crypto[rel_notnull]["dst"].isin(labeled).sum()}')

# 全部 sub_kind，兩端都有標籤
both_all = crypto[
    crypto['src'].isin(labeled) &
    rel_notnull &
    crypto['dst'].isin(labeled)
]
print(f'\n兩端都有標籤的邊 (全部 sub_kind): {len(both_all)}')

# sub_kind==1
sub1 = crypto[crypto['sub_kind'] == 1]
print(f'\nsub_kind==1 筆數: {len(sub1)}')
print(f'sub_kind==1 relation_user_id 非空: {sub1["relation_user_id"].notna().sum()}')
print('\nsub_kind==1 前5筆:')
print(sub1[['user_id','relation_user_id','kind','sub_kind']].head())

# sub_kind==0 (外部)
sub0 = crypto[crypto['sub_kind'] == 0]
print(f'\nsub_kind==0 筆數: {len(sub0)}')
print(f'sub_kind==0 user_id 在 labeled: {sub0["src"].isin(labeled).sum()}')
