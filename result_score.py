import pandas as pd
import numpy as np
from tqdm import tqdm

def load_scheme1(filepath: str) -> pd.DataFrame:
    """
    加载方案一的数据。
    session_id,item_recommendations
    200,8060 17089 17089 4758
    """
    df = pd.read_csv(filepath)
    # 将推荐字符串分割成列表，并处理潜在的空值
    df['item_recommendations'] = df['item_recommendations'].fillna('').apply(lambda x: [int(i) for i in x.split()])
    return df

def load_scheme2_as_ground_truth(filepath: str) -> dict:
    """
    加载方案二的数据，并将其转换为一个字典，作为评估的基准。
    key: session_id, value: set of recommended item_ids for fast lookup.
    """
    df = pd.read_csv(filepath)
    ground_truth = df.groupby('session_id')['item_id'].apply(set).to_dict()
    return ground_truth

def average_precision_at_k(predictions: list, ground_truth: set, k: int) -> float:
    """计算单个会话的 Average Precision @k."""
    predictions = predictions[:k]
    if not ground_truth:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predictions):
        if p in ground_truth:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # 如果基准列表为空，我们无法进行除法运算
    if not ground_truth:
         return 0.0

    return score / min(len(ground_truth), k)

def precision_recall_at_k(predictions: list, ground_truth: set, k: int) -> tuple:
    """计算单个会话的 Precision@k 和 Recall@k."""
    predictions_at_k = predictions[:k]
    
    # 使用集合运算快速找到交集
    hits = set(predictions_at_k) & ground_truth
    
    # 计算 Precision
    precision = len(hits) / k if k > 0 else 0.0
    
    # 计算 Recall
    recall = len(hits) / len(ground_truth) if ground_truth else 0.0
    
    return precision, recall

def reciprocal_rank(predictions: list, ground_truth: set) -> float:
    """计算单个会话的 Reciprocal Rank."""
    for i, p in enumerate(predictions):
        if p in ground_truth:
            return 1.0 / (i + 1.0)
    return 0.0

def evaluate_recs(scheme1_df: pd.DataFrame, ground_truth: dict, k_list: list):
    """
    在所有会话上评估推荐结果并计算平均指标。
    """
    metrics = {k: {'precision': [], 'recall': [], 'map': []} for k in k_list}
    mrr_scores = []

    # 使用tqdm显示进度条
    for _, row in tqdm(scheme1_df.iterrows(), total=len(scheme1_df), desc="Evaluating sessions"):
        session_id = row['session_id']
        predictions = row['item_recommendations']
        
        # 确保该会话在基准数据中存在
        if session_id in ground_truth:
            gt_set = ground_truth[session_id]
            
            # 计算 MRR (与 k 无关)
            mrr_scores.append(reciprocal_rank(predictions, gt_set))
            
            # 计算不同 k 值下的指标
            for k in k_list:
                precision, recall = precision_recall_at_k(predictions, gt_set, k)
                ap_at_k = average_precision_at_k(predictions, gt_set, k)
                
                metrics[k]['precision'].append(precision)
                metrics[k]['recall'].append(recall)
                metrics[k]['map'].append(ap_at_k)

    print("\n--- Recommendation Quality Evaluation ---")
    if not mrr_scores:
        print("No common sessions found between the two files. Cannot evaluate.")
        return

    print(f"\nMean Reciprocal Rank (MRR): {np.mean(mrr_scores):.4f}")
    
    for k in k_list:
        mean_precision = np.mean(metrics[k]['precision'])
        mean_recall = np.mean(metrics[k]['recall'])
        mean_ap = np.mean(metrics[k]['map'])
        
        print(f"\n--- Metrics @{k} ---")
        print(f"Precision@{k}: {mean_precision:.4f}")
        print(f"Recall@{k}:    {mean_recall:.4f}")
        print(f"MAP@{k}:       {mean_ap:.4f}")
        
if __name__ == '__main__':
    # --- 配置 ---
    # 将您的文件名放在这里
    SCHEME1_FILE = 'submission_final_filled.csv'
    SCHEME2_FILE = 'final_sub.csv'
    
    # 您想评估的 k 值列表
    K_VALUES = [10, 25, 50, 100]

    # --- 运行评估 ---
    print("Step 1: Loading Scheme 1 (predictions)...")
    scheme1_data = load_scheme1(SCHEME1_FILE)
    
    print("Step 2: Loading Scheme 2 (ground truth)...")
    ground_truth_data = load_scheme2_as_ground_truth(SCHEME2_FILE)
    
    print("Step 3: Calculating metrics...")
    evaluate_recs(scheme1_data, ground_truth_data, K_VALUES)