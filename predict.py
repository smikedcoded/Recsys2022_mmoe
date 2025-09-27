import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from mmoe_3 import MMOE # 确保能导入您的模型类

# --- 步骤 1: 定义文件路径并加载所有必需品 ---

print("--> Step 1: Defining paths and loading artifacts...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# [核心修改] 根据您的目录结构定义路径
RAW_DATA_PATH = './archive/dressipi_recsys2022/'
ARTIFACTS_PATH = './' # 您的模型、向量、预处理器都在根目录

# 定义您所有训练好的模型的路径
MODEL_PATHS = [
    ARTIFACTS_PATH + 'mmoe_model.pth',
    # 如果有更多模型，在这里添加，例如 'widedeep_model.pth'
] 

# 加载特征“原材料”和预处理器
# [核心修改] 加载并立即重塑 item_features_df
print("   -> Loading and pivoting item features...")
_item_features_df = pd.read_csv(RAW_DATA_PATH + "item_features.csv")
item_features_wide_df = _item_features_df.pivot_table(
    index='item_id',
    columns='feature_category_id',
    values='feature_value_id',
    aggfunc='first' # <--- 关键！告诉 pandas 如何处理重复项
).reset_index()

# 重命名列名，使其更清晰
item_features_wide_df.columns = ['item_id'] + [f'item_feature_{col}' for col in item_features_wide_df.columns[1:]]
# 用0填充缺失的特征值
item_features_wide_df.fillna(0, inplace=True)
del _item_features_df # 释放内存
print("   -> Item features pivoted successfully.")


# [重要] 动态计算会话特征，因为测试集的session也需要这些特征
# 我们先从训练集计算，以便应用到测试集
train_sessions_df = pd.read_csv(RAW_DATA_PATH + "train_sessions.csv")
session_features_df = train_sessions_df.groupby('session_id').agg(
    session_len=('item_id', 'count'),
    session_nunique_items=('item_id', 'nunique')
).reset_index()

# 加载DeepWalk向量作为特征
item_embeddings_df = pd.read_pickle(ARTIFACTS_PATH + 'item_deepwalk_embeddings.pkl')
embedding_cols = [f'embed_{i}' for i in range(64)]
item_embeddings_df = pd.DataFrame(item_embeddings_df.items(), columns=['item_id', 'embedding'])
item_embeddings_df[embedding_cols] = pd.DataFrame(item_embeddings_df['embedding'].tolist(), index=item_embeddings_df.index)
item_embeddings_df.drop('embedding', axis=1, inplace=True)


# 加载训练时保存的预处理器和字典
encoders = joblib.load(ARTIFACTS_PATH + 'label_encoders.pkl')
scaler = joblib.load(ARTIFACTS_PATH + 'min_max_scaler.pkl')
enc_dict = joblib.load(ARTIFACTS_PATH + 'enc_dict.pkl')

# 定义特征列名 (必须与训练时完全一致)
sparse_features = ['session_id', 'item_id'] # 已修正，不包含label
dense_features = ['session_len', 'session_nunique_items'] + [f'embed_{i}' for i in range(64)]


print("--> Step 1.5: Calculating global popular items...")
# 我们使用训练集来定义“热门”，这更公平
# value_counts() 默认就是降序排序
popular_items = train_sessions_df['item_id'].value_counts().index.tolist()
print(f"   -> Found {len(popular_items)} unique items, sorted by popularity.")

# --- 步骤 2: 创建通用的预测函数 (与之前版本基本一致) ---

def predict_scores(model, candidates_df):
    """使用加载好的模型对候选DataFrame进行评分"""
    
    pred_df = candidates_df.copy()
    
    # 1. 特征工程：合并所有需要的特征
    pred_df = pd.merge(pred_df, item_features_wide_df, on='item_id', how='left')
    pred_df = pd.merge(pred_df, session_features_df, on='session_id', how='left') # 注意：测试集的session可能不在这里面
    pred_df = pd.merge(pred_df, item_embeddings_df, on='item_id', how='left')
    pred_df.fillna(0, inplace=True)
    
    # 2. 数据预处理
    for feat in sparse_features: # 确保循环的是正确的稀疏特征列表
        if feat in encoders:
            encoder = encoders[feat]
            known_labels = set(encoder.classes_)
            
            # 识别出哪些是“新面孔” (unknown)
            is_unknown = ~pred_df[feat].isin(known_labels)
            
            # 对于认识的“老朋友”，我们正常进行 transform
            known_values = pred_df.loc[~is_unknown, feat]
            if not known_values.empty:
                pred_df.loc[~is_unknown, feat] = encoder.transform(known_values)
                
            # 对于不认识的“新面孔”，我们统一给一个默认的整数ID，比如 0
            # 这个 0 对应的是 encoder 转换后的第 0 个类别
            if is_unknown.any():
                pred_df.loc[is_unknown, feat] = 0 # Assign a default encoded value

    pred_df[dense_features] = scaler.transform(pred_df[dense_features])
    
    # 3. 转换为Tensor并预测
    feature_dict = {
        col: torch.tensor(pred_df[col].values, dtype=torch.long).to(DEVICE) for col in sparse_features
    }
    feature_dict['dense_features'] = torch.tensor(pred_df[dense_features].values, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        # [核心修改] 先接收模型返回的列表
        task_outputs = model(feature_dict)
        # [核心修改] 因为是单任务，所以用索引 [0] 取出我们需要的预测结果
        scores = task_outputs[0].squeeze(-1).cpu().numpy()
        
    return scores

# --- 步骤 3: 主程序 ---

print("--> Step 2: Loading leaderboard candidates...")
# [重要] 确保您有这个文件，或者替换成正确的测试集候选者文件名
test_candidates_df = pd.read_csv(RAW_DATA_PATH + "test_final_sessions.csv") 
# 注意：这里的示例文件是 session 文件，您需要一个包含 (session_id, item_id) 对的候选文件。
# 如果比赛方没有提供，您需要先用您的召回模型为测试集生成候选。

all_scores = []
print("--> Step 3: Starting prediction and ensembling...")
for path in MODEL_PATHS:
    # [修改点] 在实例化模型时，传入 dense_feature_cols 参数
    model = MMOE(
        enc_dict=enc_dict,
        dense_feature_cols=dense_features, # <--- 核心修改在这里！
        n_expert=3,
        embedding_dim=16,
        mmoe_hidden_dim=64,
        hidden_dim=[64]
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    scores = predict_scores(model, test_candidates_df)
    all_scores.append(scores)

test_candidates_df['final_score'] = np.mean(all_scores, axis=0)

# --- 步骤 4: 格式化并生成提交文件 ---

print("--> Step 4: Formatting and generating submission file with popular item filling...")

# [新增] 定义每个推荐列表的目标长度
TARGET_REC_LEN = 100

submission_groups = test_candidates_df.sort_values('final_score', ascending=False).groupby('session_id')

submission_data = []
for session_id, group in tqdm(submission_groups, desc="Formatting and Filling"):
    # 1. 获取模型生成的推荐列表
    recs = group['item_id'].tolist()
    
    # 2. 如果列表已经足够长，直接截断
    if len(recs) >= TARGET_REC_LEN:
        final_recs = recs[:TARGET_REC_LEN]
    else:
        # 3. 如果列表不够长，开始填充
        final_recs = recs
        # 为了快速查找，将已有推荐转为set
        recs_set = set(recs)
        # 需要填充的数量
        num_needed = TARGET_REC_LEN - len(recs)
        
        # 遍历全局热门商品列表
        for pop_item in popular_items:
            # 如果热门商品不在已有推荐中，就添加它
            if pop_item not in recs_set:
                final_recs.append(pop_item)
                # 每添加一个，所需数量就减一
                num_needed -= 1
                # 如果已经填满了，就跳出循环
                if num_needed == 0:
                    break
    
    submission_data.append([session_id, ' '.join(map(str, final_recs))])

submission_df = pd.DataFrame(submission_data, columns=['session_id', 'item_recommendations'])
submission_df.to_csv('submission_final_filled.csv', index=False)

print("\n🎉 Submission file 'submission_final_filled.csv' generated successfully!")
print("All recommendation lists now have a length of", TARGET_REC_LEN)
print(submission_df.head())