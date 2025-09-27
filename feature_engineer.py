# --- 加载所需数据 ---
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


path='./archive/dressipi_recsys2022/'
train_sessions = pd.read_csv(path+"train_sessions.csv")
train_purchase = pd.read_csv(path+"train_purchases.csv")
item_deepwalk_embeddings = pickle.load(open('./item_deepwalk_embeddings.pkl', 'rb'))

# 1. 上一步生成的样本
samples_df = pd.read_csv("mmoe_training_samples_faiss.csv")

# 2. 物品特征
item_features_df = pd.read_csv("./archive/dressipi_recsys2022/item_features.csv")

# 3. 会话/用户特征 (这里用 session 级特征举例)
# 我们可以简单计算一些会话特征
session_features = train_sessions.groupby('session_id')['item_id'].agg(['count', 'nunique']).reset_index()
session_features.rename(columns={'count': 'session_len', 'nunique': 'session_nunique_items'}, inplace=True)

# 4. DeepWalk 向量本身也是最重要的特征
item_embeddings_df = pd.DataFrame(item_deepwalk_embeddings.items(), columns=['item_id', 'embedding'])
embedding_cols = [f'embed_{i}' for i in range(64)]
item_embeddings_df[embedding_cols] = pd.DataFrame(item_embeddings_df['embedding'].tolist(), index=item_embeddings_df.index)
item_embeddings_df.drop('embedding', axis=1, inplace=True)

item_features_df['feature_full_id'] = item_features_df['feature_category_id'].astype(str) + '_' + item_features_df['feature_value_id'].astype(str)

# 1. 将每个 item_id 的所有特征聚合到一个字符串中，用空格隔开
# item_id -> "feat_1 feat_2 feat_3 ..."
item_feature_sentences = item_features_df.groupby('item_id')['feature_full_id'].apply(lambda x: ' '.join(x)).reset_index()
item_feature_sentences.rename(columns={'feature_full_id': 'feature_sentence'}, inplace=True)

print("每个物品的特征“句子”:")
print(item_feature_sentences.head())

# 2. 初始化并训练 TfidfVectorizer
# max_features 控制了最终向量的维度（即“宽表”的列数），可以有效控制内存
vectorizer = TfidfVectorizer(max_features=5000) # 例如，我们只取最重要的 5000 个特征作为列

# 3. 转换特征句子为 TF-IDF 向量（这是一个稀疏矩阵）
item_tfidf_matrix = vectorizer.fit_transform(item_feature_sentences['feature_sentence'])

# 4. 将稀疏矩阵转换为 DataFrame 以便合并
item_features_tfidf = pd.DataFrame(
    item_tfidf_matrix.toarray(), 
    index=item_feature_sentences['item_id'],
    columns=vectorizer.get_feature_names_out()
).reset_index()

print("TF-IDF 特征宽表:")
print(item_features_tfidf.head())


# --- 合并特征 ---
# 1. 合并物品特征
final_df = pd.merge(samples_df, item_features_tfidf, on='item_id', how='left')

# 2. 合并会话特征
final_df = pd.merge(final_df, session_features, on='session_id', how='left')

# 3. 合并 DeepWalk 向量特征
final_df = pd.merge(final_df, item_embeddings_df, on='item_id', how='left')

# 填充缺失值
final_df.fillna(0, inplace=True)

# 保存最终的特征文件
final_df.to_csv("final_features_for_mmoe.csv", index=False)
print("特征工程完成！")
print(final_df.head())