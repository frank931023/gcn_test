"""
数据处理模块 - 负责读取、清理和标准化交易数据
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataProcessor:
    """处理交易数据和黑名单用户的类"""
    
    def __init__(self, data_dir='data/'):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据文件目录
        """
        self.data_dir = data_dir
        self.user_info_df = None
        self.crypto_transfer_df = None
        self.train_label_df = None
        self.bad_users = set()
        self.account_to_id = {}
        self.id_to_account = {}
        
    def load_data(self):
        """读取CSV数据文件"""
        print("正在读取数据文件...")
        self.user_info_df = pd.read_csv(f'{self.data_dir}user_info.csv')
        self.crypto_transfer_df = pd.read_csv(f'{self.data_dir}crypto_transfer.csv')
        self.train_label_df = pd.read_csv(f'{self.data_dir}train_label.csv')
        print("数据文件读取完成")
        
    def extract_bad_users(self):
        """提取黑名单用户集合"""
        self.bad_users = set(
            self.train_label_df[self.train_label_df['status'] == 1]['user_id']
            .values.astype(str)
        )
        print(f"检测到 {len(self.bad_users)} 名黑名单用户")
        return self.bad_users
    
    def standardize_transactions(self):
        """
        标准化交易数据并打上洗钱标签
        
        Returns:
            pd.DataFrame: 标准化后的交易数据
        """
        print("\n开始标准化交易数据...")
        clean_records = []
        
        for index, row in tqdm(
            self.crypto_transfer_df.iterrows(), 
            total=len(self.crypto_transfer_df), 
            desc="处理交易"
        ):
            src = None
            dst = None
            
            # 金额转换 (乘以 1e-8)
            amount = float(row['ori_samount']) * 1e-8
            
            # 判断交易方向逻辑
            if row['sub_kind'] == 1:
                # 内部转账
                src = row['user_id']
                dst = row['relation_user_id']
            elif row['sub_kind'] == 0:
                # 外部提充
                if row['kind'] == 0:
                    # 入金 (存款)
                    src = row['from_wallet_hash']
                    dst = row['user_id']
                elif row['kind'] == 1:
                    # 出金 (提现)
                    src = row['user_id']
                    dst = row['to_wallet_hash']
            
            # 确保起点与终点都有效
            if pd.notna(src) and pd.notna(dst):
                src = str(src)
                dst = str(dst)
                
                # 核心标签逻辑：只要任一方是黑名单，就标记为洗钱
                is_laundering = 1 if (src in self.bad_users or dst in self.bad_users) else 0
                
                clean_records.append({
                    'src_account': src,
                    'dst_account': dst,
                    'ori_samount': amount,
                    'is_laundering': is_laundering
                })
        
        standardized_df = pd.DataFrame(clean_records)
        
        print("\n--- 标准化交易表示例 ---")
        print(standardized_df.head())
        print("\n--- 交易标签分布 ---")
        print(standardized_df['is_laundering'].value_counts())
        
        return standardized_df
    
    def create_account_mapping(self, df):
        """
        创建账户到ID的映射
        
        Args:
            df: 包含交易数据的DataFrame
            
        Returns:
            tuple: (account_to_id, id_to_account)
        """
        print("\n创建账户映射...")
        all_accounts = pd.concat([
            df['src_account'],
            df['dst_account']
        ]).unique()
        
        self.account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
        self.id_to_account = {i: acc for acc, i in self.account_to_id.items()}
        
        print(f"总共有 {len(self.account_to_id)} 个独特账户")
        
        return self.account_to_id, self.id_to_account
    
    def map_accounts_to_ids(self, df):
        """
        将账户映射到ID
        
        Args:
            df: 包含账户信息的DataFrame
            
        Returns:
            pd.DataFrame: 带有src_id和dst_id列的DataFrame
        """
        df = df.copy()
        df['src_id'] = df['src_account'].map(self.account_to_id)
        df['dst_id'] = df['dst_account'].map(self.account_to_id)
        return df
    
    @property
    def num_nodes(self):
        """获取节点总数"""
        return len(self.account_to_id)
