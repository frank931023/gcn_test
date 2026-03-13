"""
数据采样模块 - 处理类不平衡问题，使用SMOTE和NearMiss
"""
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline
import pandas as pd


class DataResampler:
    """处理数据采样和平衡的类"""
    
    def __init__(self):
        """初始化采样器"""
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_res = None
        self.y_res = None
        self.train_resampled_df = None
        
    def split_train_test(self, df, test_size=0.2, random_state=42):
        """
        按分层比例分割训练集和测试集
        
        Args:
            df: 原始DataFrame
            test_size: 测试集大小比例
            random_state: 随机种子
            
        Returns:
            tuple: (train_df, test_df)
        """
        print("\n分割训练集和测试集...")
        self.train_df, self.test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['is_laundering'],
            random_state=random_state
        )
        
        original_ratio = self.train_df['is_laundering'].mean()
        print(f"原始训练集洗钱交易比例: {original_ratio:.4f}")
        print(f"  训练集大小: {len(self.train_df)}")
        print(f"  测试集大小: {len(self.test_df)}")
        
        return self.train_df, self.test_df
    
    def resample_data(self, train_df, smote_sampling_strategy=0.1):
        """
        使用SMOTE和NearMiss重采样处理类不平衡
        
        Args:
            train_df: 训练DataFrame
            smote_sampling_strategy: SMOTE采样策略
            
        Returns:
            pd.DataFrame: 重采样后的训练数据
        """
        if train_df is None:
            train_df = self.train_df
            
        print("\n开始重采样处理...")

        
        
        # 准备X和y
        X_train = train_df[['src_account', 'dst_account', 'ori_samount']]
        y_train = train_df['is_laundering']
        
        # 创建重采样管道
        # SMOTENC: 处理混合特征（第0和第1列是类别特征）
        smote_nc = SMOTENC(
            categorical_features=[0, 1],
            random_state=42,
            sampling_strategy=smote_sampling_strategy
        )
        near_miss = NearMiss(version=1, sampling_strategy=1.0)
        
        resample_pipe = Pipeline([
            ('smotenc', smote_nc),
            ('nearmiss', near_miss)
        ])
        
        # 执行重采样
        X_res, y_res = resample_pipe.fit_resample(X_train, y_train)
        
        # 创建重采样的DataFrame
        train_resampled_df = pd.DataFrame(
            X_res, 
            columns=['src_account', 'dst_account', 'ori_samount']
        )
        train_resampled_df['is_laundering'] = y_res
        
        resampled_ratio = train_resampled_df['is_laundering'].mean()
        print(f"重采样后训练集洗钱交易比例: {resampled_ratio:.4f}")
        print(f"  重采样后训练集大小: {len(train_resampled_df)}")
        
        self.train_resampled_df = train_resampled_df
        self.X_res = X_res
        self.y_res = y_res
        
        return train_resampled_df
    
    def get_resampled_data(self):
        """获取重采样后的数据"""
        if self.train_resampled_df is None:
            raise ValueError("请先调用 resample_data() 进行重采样")
        return self.train_resampled_df
