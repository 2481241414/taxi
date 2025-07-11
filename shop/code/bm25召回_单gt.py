import pandas as pd
import os
import json
import ast
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
import time 

# --- 召回器和评测模块 ---
from rank_bm25 import BM25Okapi
import jieba

from sklearn.metrics import roc_auc_score


class ToolRetriever:
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list):
        self.tool_corpus = all_tools_corpus
        self.tool_definitions = all_tools_definitions
        # 增加jieba自定义词典，可以提高对工具名称的分词准确性
        for tool in self.tool_definitions:
            jieba.add_word(tool.get('name', ''), freq=100)
        tokenized_corpus = [self._tokenize(doc) for doc in self.tool_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    def _tokenize(self, text: str) -> list[str]: return jieba.lcut(text, cut_all=False)
    # retrieve 方法不再被主循环直接调用，但保留它以便其他地方可能使用
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = doc_scores.argsort()[-top_k:][::-1]
        return [self.tool_definitions[i] for i in top_k_indices if doc_scores[i] > 0]

# ... (所有评测函数保持不变) ...
def _get_tool_names(tools: list) -> set:
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    return len(retrieved_names_at_k.intersection(ground_truth_names)) / len(ground_truth_names)

def calculate_completeness_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    return 1.0 if ground_truth_names.issubset(retrieved_names_at_k) else 0.0

def calculate_ndcg_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, tool in enumerate(retrieved[:k]) if tool.get('name') in ground_truth_names)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_names), k)))
    return dcg / idcg if idcg > 0 else 0.0

def calculate_precision_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if k == 0: return 0.0
    retrieved_names = _get_tool_names(retrieved[:k])
    gt_names = _get_tool_names(ground_truth)
    return len(retrieved_names & gt_names) / k

def calculate_f1_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    p = calculate_precision_at_k(retrieved, ground_truth, k)
    r = calculate_recall_at_k(retrieved, ground_truth, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def calculate_hit_ratio_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    retrieved_names = _get_tool_names(retrieved[:k])
    return 1.0 if retrieved_names & _get_tool_names(ground_truth) else 0.0

def calculate_average_precision_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    gt_names = _get_tool_names(ground_truth)
    if not gt_names: return 1.0
    hit_count = 0
    sum_prec = 0.0
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            hit_count += 1
            sum_prec += hit_count / (i + 1)
    return sum_prec / len(gt_names)

def calculate_mrr_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    gt_names = _get_tool_names(ground_truth)
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            return 1.0 / (i + 1)
    return 0.0

def calculate_auc_for_query(all_scores: np.ndarray, tool_defs: list, ground_truth: list) -> float:
    gt_names = _get_tool_names(ground_truth)
    labels = [1 if t['name'] in gt_names else 0 for t in tool_defs]
    try:
        # 确保至少有一个正样本和一个负样本
        if len(set(labels)) < 2:
            return 0.5
        return roc_auc_score(labels, all_scores)
    except ValueError:
        return 0.5

# --- 权威工具定义模块 (保持不变) ---
def get_exact_tool_definitions():
    # ... (函数内容不变) ...
    tools = [
        {"name": "open_orders_bought(app, order_status)", "description": "在app应用程序中查看买入的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表，在不指定购买还是售卖的订单时，及全都要看时使用。"},
        {"name": "search_order(app, search_info, order_status)", "description": "在app应用程序中搜索订单"},
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},
        {"name": "open_cart_content(app, filter_type)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）指定类型的商品"},
        {"name": "search_cart_content(app, search_info)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）查找商品"},
        {"name": "open_customer_service(app)", "description": "在app应用程序中联系客服"},
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到，领取积分、金币等奖励的操作"},
        {"name": "open_favorite_goods(app, filter_type, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面，并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并按照条件进行筛选"},
        {"name": "search_in_favorite_goods(app, search_info)", "description": "在app程序中打开收藏的、喜爱、想要或关注商品的页面，并在其中的搜索栏中进行搜索"},
        {"name": "search_in_favorite_stores(app, search_info)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并在其中的搜索栏搜索商品"},
        {"name": "search_goods(app, search_info, order_type)", "description": "在app程序中依据名称搜索商品，可以指定搜索结果的排序方式"},
        {"name": "search_stores(app, search_info, filter_type, order_type)", "description": "在app程序中依据名称搜索店铺，可以使用筛选器限制搜索结果，也可以指定搜索结果的排序方式"},
        {"name": "open_search_history(app)", "description": "打开app程序的搜索历史界面"},
        {"name": "delete_search_history(app)", "description": "清除app中的搜索历史"},
        {"name": "open_camera_search(app)", "description": "打开app程序的图片搜索功能"},
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools

# --- 主程序 ---
def main():
    # --- 0. 配置区域 ---
    annotated_data_file_path = r'/home/workspace/lgq/shop/data/单gt.csv' # 假设文件在当前目录
    K_VALUES = [1, 2, 3, 4, 5]
    NUM_EXAMPLES_TO_PRINT = 1

    # --- 1. 数据加载 ---
    print("--- 步骤 1: 加载已标注数据并划分 ---")
    try:
        required_columns = ['instruction_template', 'final_query', 'is_train', 'available_tools']
        data_df = pd.read_csv(annotated_data_file_path, usecols=required_columns)
    except Exception as e:
        print(f"错误: 读取文件 '{annotated_data_file_path}' 失败. {e}")
        return
    
    def parse_tools(tool_string):
        try: return ast.literal_eval(tool_string)
        except (ValueError, SyntaxError): return []
    data_df['available_tools'] = data_df['available_tools'].apply(parse_tools)
    
    train_df = data_df[data_df['is_train'] == 0].copy()
    test_df = data_df[data_df['is_train'] == 1].copy()
    print(f"数据划分完成：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条。\n")


    # --- 2. 构建增强的 BM25 语料库 (仅使用训练集) ---
    print("--- 步骤 2: 构建增强的 BM25 语料库 (仅使用训练集) ---")
    
    tool_text_aggregator = defaultdict(list)
    for _, row in train_df.iterrows():
        if row['available_tools']:
            tool_name = row['available_tools'][0]['name']
            tool_text_aggregator[tool_name].append(row['instruction_template'])
            tool_text_aggregator[tool_name].append(row['final_query'])

    all_tools_corpus = []
    all_tools_definitions = get_exact_tool_definitions()
    
    for tool_def in all_tools_definitions:
        tool_name = tool_def['name']
        aggregated_text = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
        document = f"这是一个工具，它的功能是 {tool_def['description']}。用户可能会这样说：{aggregated_text}"
        all_tools_corpus.append(document)

    print(f"语料库构建完成，共 {len(all_tools_corpus)} 个工具。\n")

    # --- 3. 构建召回器 ---
    print("--- 步骤 3: 构建 BM25 召回器 ---")
    retriever = ToolRetriever(all_tools_corpus, all_tools_definitions)
    print("召回器构建完成。\n")

    # --- 4. 在测试集上评测 ---
    print(f"--- 步骤 4: 开始在 {len(test_df)} 个测试集样本上进行评测 ---")
    
    eval_df = test_df.copy()

    results = {
        'Recall@K':    {k: [] for k in K_VALUES},
        'HR@K':        {k: [] for k in K_VALUES},
        'MAP@K':       {k: [] for k in K_VALUES},
        'MRR@K':       {k: [] for k in K_VALUES},
        'NDCG@K':      {k: [] for k in K_VALUES},
        'COMP@K':      {k: [] for k in K_VALUES},
        'AUC':         [],
        'processing_time': []  # 用于存储每个样本的运行时间
    }

    for i, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="测试集评测中")):
        query = row['final_query']
        ground_truth = row['available_tools'] 
        
        # --- 核心召回逻辑 (计时部分) ---
        start_time = time.perf_counter()

        # 优化：只计算一次分数，然后复用
        tokenized_query = retriever._tokenize(query)
        all_scores = retriever.bm25.get_scores(tokenized_query)
        
        # 从全量分数中获取 top_k 结果，避免重复调用 retrieve 方法
        max_k = max(K_VALUES)
        # 获取分数最高的 top_k 个工具的索引
        top_k_indices = all_scores.argsort()[-max_k:][::-1]
        # 根据索引构建召回列表，并过滤掉分数为0的结果
        retrieved = [retriever.tool_definitions[i] for i in top_k_indices if all_scores[i] > 0]
        
        end_time = time.perf_counter()
        # --- 计时结束 ---

        # 记录本次运行时间
        duration = end_time - start_time
        results['processing_time'].append(duration)
        
        # **计算 AUC (使用已计算好的 all_scores)**
        results['AUC'].append(calculate_auc_for_query(all_scores, all_tools_definitions, ground_truth))
        
        # 打印示例（逻辑不变）
        if i < NUM_EXAMPLES_TO_PRINT:
            print(f"\n\n==================== 测试样本 {i+1} (耗时: {duration*1000:.2f} ms) ====================")
            print(f" [查询 Query]: {query}")
            gt_names = [t.get('name', 'N/A') for t in ground_truth]
            print(f" [真实工具 Ground Truth (唯一)]: {gt_names}")
            retrieved_names = [t.get('name', 'N/A') for t in retrieved]
            print(f" [召回结果 Prediction @{max_k}]: {retrieved_names}")
            is_complete = calculate_completeness_at_k(retrieved, ground_truth, k=max_k)
            status = "✅ 命中 (Hit)" if is_complete else "❌ 偏离 (Miss)"
            print(f" [评测状态 @{max_k}]: {status}")
            print("===================================================================================")
            
        # **针对每个 K 计算所有指标 (使用已获取的 retrieved 列表)**
        for k in K_VALUES:
            results['Recall@K'][k].append(calculate_recall_at_k(retrieved, ground_truth, k))
            results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved, ground_truth, k))
            results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved, ground_truth, k))
            results['MRR@K'][k].append(calculate_mrr_at_k(retrieved, ground_truth, k))
            results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved, ground_truth, k))
            results['COMP@K'][k].append(calculate_completeness_at_k(retrieved, ground_truth, k))
            
        
    # --- 5. 汇总并报告结果 ---
    print("\n\n--- 步骤 5: 评测结果报告 (基于测试集和精确GT) ---")
    final_scores = {}
    for metric, vals in results.items():
        if metric == 'AUC':
            final_scores['AUC'] = np.mean(vals)
        elif metric == 'processing_time':
             continue # 时间单独处理
        else:
            final_scores[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
            
    # 构建指标报告表格
    report_df = pd.DataFrame({
        m: final_scores[m] for m in [
            'Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K'
        ]
    }).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    print("BM25 召回模型在测试集上的评测结果:")
    print("-" * 50)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores['AUC']:.4f}")
    print("-" * 50)

    #  4. 报告性能统计数据 
    total_time = np.sum(results['processing_time'])
    avg_time_ms = np.mean(results['processing_time']) * 1000  # 转换为毫秒
    num_samples = len(eval_df)
    qps = num_samples / total_time if total_time > 0 else 0

    print("\n性能评测:")
    print("-" * 50)
    print(f"测试样本总数: {num_samples} 条")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"平均每条样本耗时: {avg_time_ms:.4f} 毫秒")
    print(f"每秒查询率 (QPS): {qps:.2f}")
    print("-" * 50)


if __name__ == "__main__":
    main()