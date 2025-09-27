import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity # <-- 这个库现在可以不用了
import faiss # [新增] 导入 faiss 库
import time
import random

print("="*50)
print("▶️ 脚本开始运行...")
start_time = time.time()

# --- 加载所需数据 (这部分不变) ---
print("\n🔄 [阶段 1/5] 正在加载数据...")
# 1. DeepWalk 产出的物品向量
item_deepwalk_embeddings = pickle.load(open('./item_deepwalk_embeddings.pkl', 'rb'))
item_list = list(item_deepwalk_embeddings.keys())
embedding_matrix = np.array(list(item_deepwalk_embeddings.values()))

# 2. 训练数据中的真实购买记录 (ground truth)
train_purchases = pd.read_csv("./archive/dressipi_recsys2022/train_purchases.csv")

# 3. 训练会话数据，用来找到召回的锚点 (last_item)
train_sessions = pd.read_csv("./archive/dressipi_recsys2022/train_sessions.csv")
print("✅ [阶段 1/5] 数据加载完毕。")


# --- 数据预处理 (这部分不变) ---
print("\n🔄 [阶段 2/5] 正在预处理 session 数据...")

# 结合 format 和 errors 参数
# 当日期格式混杂时使用
train_sessions['date'] = pd.to_datetime(train_sessions['date'], format='mixed')

last_items_df = train_sessions.loc[train_sessions.groupby('session_id')['date'].idxmax()]
print("✅ [阶段 2/5] Session 数据预处理完毕。")


# --- [新增] Faiss 索引构建部分 ---
# 在主循环开始前，我们先为所有的物品向量构建一个高效的 ANN 索引
print("\n🔄 [阶段 3/5] 正在构建 Faiss 索引...")
# L2 距离和余弦相似度在归一化向量上是等价的，所以我们先进行归一化
embedding_matrix_normalized = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)

# 获取向量维度
d = embedding_matrix_normalized.shape[1] 
# 构建索引，IndexFlatL2 是最基础的 L2 距离索引
index = faiss.IndexFlatL2(d)
# Faiss 需要 float32 类型的数据
index.add(embedding_matrix_normalized.astype('float32')) 
print(f"✅ Faiss 索引构建完毕，共索引了 {index.ntotal} 个向量。")


# --- 开始生成训练样本 ---
print("\n🔄 [阶段 4/5] 开始生成训练样本 (使用 Faiss 进行高效召回)...")
K = 100 # 每条记录召回100个候选
NEG_SAMPLE_RATIO = 4 # 负采样比例，每个正样本配4个负样本
training_samples = []

for _, row in tqdm(last_items_df.iterrows(), total=len(last_items_df), desc="生成样本中"):
    session_id = row['session_id']
    last_item_id = row['item_id']

    if last_item_id not in item_deepwalk_embeddings:
        continue

    # 1. 使用 Faiss 高效召回 Top K
    anchor_vector = item_deepwalk_embeddings[last_item_id].reshape(1, -1)
    anchor_vector_normalized = anchor_vector / np.linalg.norm(anchor_vector)
    D, I = index.search(anchor_vector_normalized.astype('float32'), K)
    candidate_items = [item_list[i] for i in I[0]]

    # 2. 打标签并进行负采样
    purchase_series = train_purchases.loc[train_purchases['session_id'] == session_id, 'item_id']
    if not purchase_series.empty:
        true_purchase_item = purchase_series.iloc[0]

        # 确保正样本在候选集中，并加入样本列表
        if true_purchase_item in candidate_items:
            training_samples.append([session_id, true_purchase_item, 1])
            
            # 从候选集中移除正样本，得到纯负样本列表
            negative_candidates = [item for item in candidate_items if item != true_purchase_item]
            
            # [核心修改] 从负样本列表中随机抽取指定数量的样本
            num_neg_samples = min(len(negative_candidates), NEG_SAMPLE_RATIO)
            sampled_negatives = random.sample(negative_candidates, num_neg_samples)
            
            for neg_item in sampled_negatives:
                training_samples.append([session_id, neg_item, 0])

            
print("✅ [阶段 4/5] 训练样本生成完毕。")


# --- 后处理与保存 (这部分不变) ---
print("\n🔄 [阶段 5/5] 正在处理并保存最终样本...")
training_df = pd.DataFrame(training_samples, columns=['session_id', 'item_id', 'label'])
print("\n--- 样本预览 ---")
print(training_df.head())
print(f"正样本比例: {training_df['label'].mean():.4f}")
print("----------------")
training_df.to_csv("mmoe_training_samples_faiss.csv", index=False)
print("✅ [阶段 5/5] 文件保存成功！")


end_time = time.time()
print(f"\n🎉 脚本全部运行完毕！总耗时: {end_time - start_time:.2f} 秒。")
print("="*50)