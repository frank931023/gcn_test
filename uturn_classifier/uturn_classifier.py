import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('uturn_classifier_results', exist_ok=True)

print("=" * 80)
print("U 型迴轉特徵分類模型訓練")
print("=" * 80)

# 讀取數據
print("\n[1/7] 讀取數據...")
twd_transfer = pd.read_csv('data/twd_transfer.csv')
usdt_trading = pd.read_csv('data/usdt_twd_trading.csv')
crypto_transfer = pd.read_csv('data/crypto_transfer.csv')
train_label = pd.read_csv('data/train_label.csv')

# 轉換時間欄位
twd_transfer['created_at'] = pd.to_datetime(twd_transfer['created_at'])
usdt_trading['updated_at'] = pd.to_datetime(usdt_trading['updated_at'])
crypto_transfer['created_at'] = pd.to_datetime(crypto_transfer['created_at'])

print(f"TWD Transfer: {len(twd_transfer)} 筆")
print(f"USDT Trading: {len(usdt_trading)} 筆")
print(f"Crypto Transfer: {len(crypto_transfer)} 筆")

# 建立用戶標籤字典
user_labels = dict(zip(train_label['user_id'], train_label['status']))

def calculate_balance_timeline(user_id):
    """計算用戶的餘額時間線"""
    user_twd = twd_transfer[twd_transfer['user_id'] == user_id].copy()
    user_usdt = usdt_trading[usdt_trading['user_id'] == user_id].copy()
    user_crypto = crypto_transfer[crypto_transfer['user_id'] == user_id].copy()
    
    events = []
    
    # TWD 加值/提領
    for _, row in user_twd.iterrows():
        amount = row['ori_samount'] / 1e8
        if row['kind'] == 0:
            events.append({'time': row['created_at'], 'type': 'twd_deposit', 
                          'twd_change': amount, 'usdt_change': 0})
        else:
            events.append({'time': row['created_at'], 'type': 'twd_withdraw', 
                          'twd_change': -amount, 'usdt_change': 0})
    
    # USDT 交易
    for _, row in user_usdt.iterrows():
        usdt_amount = row['trade_samount'] / 1e8
        twd_rate = row['twd_srate'] / 1e8
        twd_amount = usdt_amount * twd_rate
        
        if row['is_buy'] == 1:
            events.append({'time': row['updated_at'], 'type': 'buy_usdt',
                          'twd_change': -twd_amount, 'usdt_change': usdt_amount})
        else:
            events.append({'time': row['updated_at'], 'type': 'sell_usdt',
                          'twd_change': twd_amount, 'usdt_change': -usdt_amount})
    
    # Crypto 提領
    for _, row in user_crypto.iterrows():
        if row['kind'] == 1:
            crypto_amount = row['ori_samount'] / 1e8
            twd_rate = row['twd_srate'] / 1e8 if pd.notna(row['twd_srate']) else 30
            twd_value = crypto_amount * twd_rate
            events.append({'time': row['created_at'], 'type': 'crypto_withdraw',
                          'twd_change': 0, 'usdt_change': -twd_value / 30})
    
    if not events:
        return None
    
    events_df = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
    events_df['twd_balance'] = events_df['twd_change'].cumsum()
    events_df['usdt_balance'] = events_df['usdt_change'].cumsum()
    events_df['total_balance_twd'] = events_df['twd_balance'] + events_df['usdt_balance'] * 30
    
    return events_df

def extract_uturn_features(events_df):
    """提取 U 型迴轉特徵（只提取脈衝次數和 U 型迴轉比率）"""
    if events_df is None or len(events_df) == 0:
        return {'uturn_ratio': 0, 'pulse_count': 0}
    
    zero_threshold = 1000
    
    # U 型迴轉比率
    large_transactions = events_df[abs(events_df['twd_change'] + events_df['usdt_change'] * 30) > 10000]
    uturn_count = 0
    
    for idx in large_transactions.index:
        next_events = events_df[events_df['time'] > events_df.loc[idx, 'time']]
        next_events = next_events[next_events['time'] <= events_df.loc[idx, 'time'] + timedelta(hours=24)]
        
        if len(next_events) > 0 and next_events['total_balance_twd'].min() < zero_threshold:
            uturn_count += 1
    
    uturn_ratio = uturn_count / len(events_df) if len(events_df) > 0 else 0
    
    # 脈衝次數
    pulse_count = 0
    for i in range(1, len(events_df) - 1):
        prev_balance = events_df.iloc[i-1]['total_balance_twd']
        curr_balance = events_df.iloc[i]['total_balance_twd']
        next_balance = events_df.iloc[i+1]['total_balance_twd']
        
        if prev_balance < zero_threshold and curr_balance > zero_threshold * 10 and next_balance < zero_threshold:
            pulse_count += 1
    
    return {'uturn_ratio': uturn_ratio, 'pulse_count': pulse_count}

# 提取所有用戶的特徵
print("\n[2/7] 提取特徵...")
# 只處理有標籤的用戶
labeled_users = set(train_label['user_id'])
print(f"總共 {len(labeled_users)} 個有標籤的用戶")

features_list = []
for idx, user_id in enumerate(labeled_users):
    if idx % 1000 == 0:
        print(f"  處理進度: {idx}/{len(labeled_users)}")
    
    events_df = calculate_balance_timeline(user_id)
    features = extract_uturn_features(events_df)
    features['user_id'] = user_id
    features['label'] = user_labels.get(user_id, 0)
    features_list.append(features)

features_df = pd.DataFrame(features_list)

# 移除沒有交易的用戶
features_df = features_df[(features_df['uturn_ratio'] > 0) | (features_df['pulse_count'] > 0)]

print(f"\n特徵提取完成: {len(features_df)} 個用戶")
print(f"  正常用戶: {len(features_df[features_df['label'] == 0])}")
print(f"  黑名單用戶: {len(features_df[features_df['label'] == 1])}")

# 保存特徵
features_df.to_csv('uturn_classifier_results/features.csv', index=False)

# 準備訓練數據
print("\n[3/7] 準備訓練數據...")
X = features_df[['pulse_count', 'uturn_ratio']].values
y = features_df['label'].values

print(f"特徵矩陣形狀: {X.shape}")
print(f"標籤分佈: 正常={sum(y==0)}, 黑名單={sum(y==1)}")

# 切割訓練集和測試集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n訓練集: {len(X_train)} 個樣本 (正常={sum(y_train==0)}, 黑名單={sum(y_train==1)})")
print(f"測試集: {len(X_test)} 個樣本 (正常={sum(y_test==0)}, 黑名單={sum(y_test==1)})")

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定義模型
print("\n[4/7] 訓練多個模型...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=sum(y_train==0)/sum(y_train==1)),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=-1)
}

results = []

for name, model in models.items():
    print(f"\n訓練 {name}...")
    
    # 訓練模型
    if name in ['Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 計算指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 交叉驗證
    if name in ['Logistic Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV F1 Mean': cv_scores.mean(),
        'CV F1 Std': cv_scores.std()
    })
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\n[5/7] 模型比較結果:")
print(results_df.to_string(index=False))

# 選擇最佳模型
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n最佳模型: {best_model_name}")
print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")

# 使用最佳模型進行預測
if best_model_name in ['Logistic Regression']:
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

# 生成詳細報告
print("\n[6/7] 生成詳細報告...")

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred_best)
print("\n混淆矩陣:")
print(cm)

# 分類報告
print("\n分類報告:")
print(classification_report(y_test, y_pred_best, target_names=['正常用戶', '黑名單用戶']))

# 視覺化
print("\n[7/7] 生成視覺化圖表...")

# 1. 模型比較圖
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
ax = axes[0, 0]
bars = ax.barh(results_df['Model'], results_df['Accuracy'], color='skyblue', edgecolor='black')
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('模型準確率比較', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
            ha='left', va='center', fontsize=10)

# F1-Score
ax = axes[0, 1]
bars = ax.barh(results_df['Model'], results_df['F1-Score'], color='lightcoral', edgecolor='black')
ax.set_xlabel('F1-Score', fontsize=12)
ax.set_title('模型 F1-Score 比較', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
            ha='left', va='center', fontsize=10)

# Precision vs Recall
ax = axes[1, 0]
ax.scatter(results_df['Recall'], results_df['Precision'], s=200, alpha=0.6, 
           c=range(len(results_df)), cmap='viridis', edgecolors='black', linewidths=2)
for i, model in enumerate(results_df['Model']):
    ax.annotate(model, (results_df.iloc[i]['Recall'], results_df.iloc[i]['Precision']),
                fontsize=9, ha='center')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ROC-AUC
ax = axes[1, 1]
bars = ax.barh(results_df['Model'], results_df['ROC-AUC'], color='lightgreen', edgecolor='black')
ax.set_xlabel('ROC-AUC', fontsize=12)
ax.set_title('模型 ROC-AUC 比較', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
            ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('uturn_classifier_results/model_comparison.png', dpi=300, bbox_inches='tight')
print("模型比較圖已保存")

# 2. 混淆矩陣熱圖
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=['正常用戶', '黑名單用戶'],
            yticklabels=['正常用戶', '黑名單用戶'],
            ax=ax, annot_kws={'size': 16})
ax.set_xlabel('預測標籤', fontsize=12)
ax.set_ylabel('真實標籤', fontsize=12)
ax.set_title(f'混淆矩陣 - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('uturn_classifier_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩陣圖已保存")

# 3. ROC 曲線
fig, ax = plt.subplots(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
roc_auc = roc_auc_score(y_test, y_pred_proba_best)

ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC 曲線 - {best_model_name}', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uturn_classifier_results/roc_curve.png', dpi=300, bbox_inches='tight')
print("ROC 曲線已保存")

# 4. Precision-Recall 曲線
fig, ax = plt.subplots(figsize=(10, 8))
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_best)

ax.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'Precision-Recall 曲線 - {best_model_name}', fontsize=14, fontweight='bold')
ax.legend(loc="lower left", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uturn_classifier_results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("Precision-Recall 曲線已保存")

# 5. 特徵散布圖（標示預測結果）
fig, ax = plt.subplots(figsize=(12, 10))

# 真陽性 (TP)
tp_mask = (y_test == 1) & (y_pred_best == 1)
ax.scatter(X_test[tp_mask, 0], X_test[tp_mask, 1], c='red', s=100, 
           label=f'真陽性 (TP) n={sum(tp_mask)}', alpha=0.7, edgecolors='black', linewidths=1.5)

# 真陰性 (TN)
tn_mask = (y_test == 0) & (y_pred_best == 0)
ax.scatter(X_test[tn_mask, 0], X_test[tn_mask, 1], c='blue', s=50, 
           label=f'真陰性 (TN) n={sum(tn_mask)}', alpha=0.5, edgecolors='black', linewidths=0.5)

# 假陽性 (FP)
fp_mask = (y_test == 0) & (y_pred_best == 1)
ax.scatter(X_test[fp_mask, 0], X_test[fp_mask, 1], c='orange', s=150, marker='x',
           label=f'假陽性 (FP) n={sum(fp_mask)}', alpha=0.9, linewidths=3)

# 假陰性 (FN)
fn_mask = (y_test == 1) & (y_pred_best == 0)
ax.scatter(X_test[fn_mask, 0], X_test[fn_mask, 1], c='purple', s=150, marker='s',
           label=f'假陰性 (FN) n={sum(fn_mask)}', alpha=0.9, edgecolors='black', linewidths=2)

ax.set_xlabel('脈衝次數 (Pulse Count)', fontsize=12)
ax.set_ylabel('U 型迴轉比率 (U-Turn Ratio)', fontsize=12)
ax.set_title(f'預測結果散布圖 - {best_model_name}', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uturn_classifier_results/prediction_scatter.png', dpi=300, bbox_inches='tight')
print("預測結果散布圖已保存")

# 保存結果
results_df.to_csv('uturn_classifier_results/model_comparison.csv', index=False)

print("\n" + "=" * 80)
print("訓練完成！")
print("=" * 80)
print(f"\n最佳模型: {best_model_name}")
print(f"測試集 F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
print(f"測試集 ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
print(f"\n所有結果已保存到: uturn_classifier_results/")
