import pandas as pd
import ast
import numpy as np

# --- 这是您提供的核心计算逻辑 ---

def dict_list_to_set(data_list):
    """
    将列表转换为集合。根据您的数据格式，这似乎只是简单地将列表转换为集合。
    如果列表中的元素是字典，则需要更复杂的逻辑，但您的示例数据显示它们是字符串。
    """
    if not isinstance(data_list, list):
        return set()
    return set(str(item) for item in data_list)

def get_prec_and_rec(ans, ground):
    """
    根据给定的逻辑计算精确率、召回率等指标。
    ans: 模型预测结果的字符串 (e.g., "['0', '1']")
    ground: 标准答案的字符串 (e.g., "['0', '1', '2']")
    """
    # 将NaN值视为空字符串
    ans = '' if pd.isna(ans) else ans
    ground = '' if pd.isna(ground) else ground

    # 尝试将字符串解析为Python列表
    try:
        # an和ground不为空且不为'nan'字符串时才进行解析
        ans_list = ast.literal_eval(ans) if ans and ans.lower() != 'nan' else []
        ground_list = ast.literal_eval(ground) if ground and ground.lower() != 'nan' else []
    except (ValueError, SyntaxError, TypeError):
        # 如果解析失败（例如格式不正确或为空），则视为空列表
        ans_list, ground_list = [], []

    ans_set = dict_list_to_set(ans_list)
    ground_set = dict_list_to_set(ground_list)
    
    intersect = len(ans_set & ground_set)

    # 情况 1: 完全都为空，视为完全正确
    if len(ans_set) == 0 and len(ground_set) == 0:
        return {'prec': 1.0, 'rec': 1.0}

    # 情况 2: 只要有一方为空（但非两者皆空），都视为0分
    if len(ans_set) == 0 or len(ground_set) == 0:
        return {'prec': 0.0, 'rec': 0.0}

    # 情况 3: 有误选（多选），精确率直接为0
    if len(ans_set - ground_set) > 0:
        rec = intersect / len(ground_set)
        return {'prec': 0.0, 'rec': rec}
        
    # 情况 4: 没有误选 (ans_set 是 ground_set 的子集)
    prec = 1.0
    rec = intersect / len(ground_set) # 因为是子集, intersect == len(ans_set)
    
    return {'prec': prec, 'rec': rec}

def process_csv_file(input_filename, output_filename):
    """
    读取CSV文件，计算指标，并保存到新的CSV文件。
    """
    try:
        print(f"正在处理文件: {input_filename}...")
        df = pd.read_csv(input_filename)
        
        # 定义用于比较的列名
        ans_col = '模型2返回结果'
        ground_col = '模型2正确结果'

        # 检查必需的列是否存在
        if ans_col not in df.columns or ground_col not in df.columns:
            print(f"错误: 文件 '{input_filename}' 中缺少必需的列 '{ans_col}' 或 '{ground_col}'。")
            return

        # 使用apply函数逐行计算指标
        # df.apply返回一个Series，其中每个元素都是一个字典 {'prec': x, 'rec': y}
        metrics_series = df.apply(
            lambda row: get_prec_and_rec(row[ans_col], row[ground_col]),
            axis=1
        )

        # 将包含字典的Series转换为一个DataFrame
        metrics_df = metrics_series.apply(pd.Series)
        
        # 重命名新列
        metrics_df = metrics_df.rename(columns={'prec': 'precision', 'rec': 'recall'})

        # 将新计算出的列合并回原始DataFrame
        output_df = pd.concat([df, metrics_df[['precision', 'recall']]], axis=1)

        # 保存到新的CSV文件，使用 utf-8-sig 编码以确保Excel能正确打开中文
        output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"处理完成！结果已保存到: {output_filename}")

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{input_filename}'。请确保文件存在于脚本所在的目录中。")
    except Exception as e:
        print(f"处理文件 '{input_filename}' 时发生未知错误: {e}")


# --- 主程序 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    single_turn_input = 'single_turn.csv'
    single_turn_output = 'single_turn_with_metrics.csv'
    
    multi_turn_input = 'multi_turn.csv'
    multi_turn_output = 'multi_turn_with_metrics.csv'

    # 处理两个文件
    process_csv_file(single_turn_input, single_turn_output)
    print("-" * 30)
    process_csv_file(multi_turn_input, multi_turn_output)