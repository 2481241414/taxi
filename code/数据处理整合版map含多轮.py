import os
import re
import json
import pandas as pd
import time
import ast

# --- 0. Unified Instruction & Constants ---
UNIFIED_INSTRUCTION = (
    "你是一个打车助手，你能根据用户的query去选择车型大类和车型小类的组合。\n"
    "用户会给你一个query和可选车型组合。\n"
    "你要根据用户的query，只能从可选车型组合中选择合适的。\n"
    "输出格式如下：\n"
    "[{大类1:小类1}, {大类2:小类2}]\n\n"
    "特殊情况说明：\n"
    "如果用户对车型没有明确要求，则车型组合字段为空。\n"
    "如果用户要求打所有车型，则输出所有可选的车型组合。\n"
    "当用户要求的车在可选列表中没有时，则输出车型组合:[]。\n"
    "注意：只能从可选择的车型组合中选择，不能自己造组合。\n"
    "忽略用户query中提到的价格，避免因为价格去选择错误的车型。\n"
    "确保最终的输出结果不重不漏，做到精确率和召回率均达到100%。\n"
)

# --- 定义所有已知格式的特征列名 ---
# 格式1: 旧版JSON格式 (第一版代码处理的)
F1_VEHICLE_COL = '车列表'
F1_GT_COL = '模型2标准答案' # Ground Truth
F1_MULTI_Q_COLS = ['多轮对话query1', '多轮对话query2', '多轮对话query3']
F1_SINGLE_Q_CANDIDATES = ['车型理解', '价格推理', '组合场景', 'appName场景']

# 格式2: 新版Markdown多轮格式 (第二版代码处理的)
F2_Q_COLS = ['query1', 'query2', 'query3']
F2_VEHICLE_COL = '车辆候选列表'
F2_GT_CAT_COL = '大类'
F2_GT_MODE_COL = '小类'

# 格式3: 新版Markdown单轮多列格式 (刚刚提供的)
F3_Q_COL = 'query'
F3_VEHICLE_COL = '车辆候选列表'
F3_GT_CAT_COL = 'category正确答案'
F3_GT_MODE_COL = 'mode正确答案'


# --- 1. Universal Parsing and Helper Functions ---

def parse_vehicle_pairs_markdown(markdown_str):
    if not isinstance(markdown_str, str): return []
    lines = markdown_str.strip().split('\n')
    pair_set = set()
    for line in lines:
        if not line.strip().startswith('|') or '---' in line or '车型大类' in line:
            continue
        fields = [field.strip() for field in line.split('|') if field.strip()]
        if len(fields) >= 4:
            # 兼容列名不一致的情况，通常大类和小类在第3、4列
            category, mode = fields[2], fields[3]
            if category and mode:
                pair_set.add((category, mode))
    return [{cat: mode} for cat, mode in sorted(list(pair_set))]

def parse_vehicle_pairs_json(json_like_str):
    if not isinstance(json_like_str, str): return []
    match = re.search(r'["\']taxiSkuList["\']\s*:\s*(\[.*?\])', json_like_str, re.DOTALL)
    if not match: return []
    try:
        vehicle_list = json.loads(match.group(1))
        pair_set = set()
        for item in vehicle_list:
            category = item.get('category', '').strip()
            car_model = item.get('carModel', '').strip()
            if category and car_model:
                pair_set.add((category, car_model))
        return [{cat: mode} for cat, mode in sorted(list(pair_set))]
    except json.JSONDecodeError:
        return []

def parse_vehicle_pairs_auto(cell_content):
    if not isinstance(cell_content, str): return []
    if cell_content.strip().startswith('|') and cell_content.count('|') > 3:
        return parse_vehicle_pairs_markdown(cell_content)
    elif '"taxiSkuList"' in cell_content or "'taxiSkuList'" in cell_content:
        return parse_vehicle_pairs_json(cell_content)
    return []

def dedup_by_input(sft_list):
    seen, result = set(), []
    for item in sft_list:
        if item['input'] not in seen:
            seen.add(item['input'])
            result.append(item)
    return result

def build_input_query(history, query, vehicle_pairs):
    vehicle_pairs_str = json.dumps(vehicle_pairs, ensure_ascii=False)
    if history:
        return f"# 用户历史会话：\n{history}\n\n# 用户需求：\n{query}\n\n用户可以选择的车型组合为：{vehicle_pairs_str}"
    else:
        return f"用户的query是：{query}\n\n用户可以选择的车型组合为：{vehicle_pairs_str}"

# --- 2. Data Normalization Stage ---

def normalize_dataframe(df):
    """
    核心函数：检测数据格式并将其标准化为统一的中间列。
    标准中间列: _query, _history, _vehicle_pairs, _gt_category, _gt_mode
    """
    # --- Format Detection ---
    cols = df.columns
    data_format = None
    if F3_Q_COL in cols and F3_GT_CAT_COL in cols and F3_GT_MODE_COL in cols:
        data_format = 'F3'
        print("检测到格式: 新版Markdown单轮多列格式 (F3)")
    elif all(c in cols for c in F2_Q_COLS) and F2_GT_CAT_COL in cols:
        data_format = 'F2'
        print("检测到格式: 新版Markdown多轮格式 (F2)")
    elif F1_VEHICLE_COL in cols and F1_GT_COL in cols:
        data_format = 'F1'
        print("检测到格式: 旧版JSON格式 (F1)")
    else:
        raise ValueError("无法识别的CSV文件格式，缺少关键列。")

    # --- Data Normalization ---
    if data_format == 'F1':
        # 旧版JSON格式
        df['_vehicle_pairs'] = df[F1_VEHICLE_COL].apply(parse_vehicle_pairs_auto)
        
        is_multi = all(c in cols for c in F1_MULTI_Q_COLS)
        if is_multi:
            df['_history'] = df.apply(lambda r: "\n".join([f"{i+1}. {r[c].strip()}" for i, c in enumerate(F1_MULTI_Q_COLS[:-1]) if pd.notna(r[c]) and r[c].strip()]), axis=1)
            df['_query'] = df[F1_MULTI_Q_COLS[-1]]
        else: # 单轮
            found_q_col = next((c for c in F1_SINGLE_Q_CANDIDATES if c in cols), None)
            df['_history'] = ''
            df['_query'] = df[found_q_col] if found_q_col else ''
        
        # Ground truth需要从索引反查，最复杂的部分
        def gt_from_indices(row):
            indices_str = row[F1_GT_COL]
            vehicle_list_str = row[F1_VEHICLE_COL]
            try:
                indices = ast.literal_eval(str(indices_str))
            except:
                return pd.Series(['', ''])

            # 从原始JSON中构建一个临时的index->item映射
            all_vehicles = parse_vehicle_pairs_json(vehicle_list_str)
            vehicle_map = {item.get('index'): item for item in json.loads(re.search(r'(\[.*?\])', vehicle_list_str).group(1))}
            
            cats, modes = set(), set()
            for idx in indices:
                item = vehicle_map.get(str(idx))
                if item:
                    cats.add(item['category'])
                    modes.add(item['carModel'])
            return pd.Series([",".join(sorted(list(cats))), ",".join(sorted(list(modes)))])

        df[['_gt_category', '_gt_mode']] = df.apply(gt_from_indices, axis=1)

    elif data_format == 'F2':
        # 新版Markdown多轮格式
        df['_vehicle_pairs'] = df[F2_VEHICLE_COL].apply(parse_vehicle_pairs_auto)
        df['_history'] = df.apply(lambda r: "\n".join([f"{i+1}. {r[c].strip()}" for i, c in enumerate(F2_Q_COLS[:-1]) if pd.notna(r[c]) and r[c].strip()]), axis=1)
        df['_query'] = df[F2_Q_COLS[-1]]
        df['_gt_category'] = df[F2_GT_CAT_COL]
        df['_gt_mode'] = df[F2_GT_MODE_COL]
        
    elif data_format == 'F3':
        # 新版Markdown单轮多列格式
        df['_vehicle_pairs'] = df[F3_VEHICLE_COL].apply(parse_vehicle_pairs_auto)
        df['_history'] = '' # 单轮无历史
        df['_query'] = df[F3_Q_COL]
        df['_gt_category'] = df[F3_GT_CAT_COL]
        df['_gt_mode'] = df[F3_GT_MODE_COL]

    return df


# --- 3. SFT Generation Stage ---

def get_final_pairs(gt_cat, gt_mode, all_pairs):
    """根据标准化的GT和所有可用组合，筛选出最终结果"""
    def split_multi(s):
        if not s or pd.isna(s) or str(s).lower() in ['nan', '无', '']:
            return []
        return [x.strip() for x in re.split(r'[，,、]', str(s)) if x.strip()]

    cats = split_multi(gt_cat)
    modes = split_multi(gt_mode)

    if any(str(c).lower() == 'all' for c in cats) or any(str(m).lower() == 'all' for m in modes):
        return all_pairs
    if not cats and not modes:
        return []

    valid = []
    for pair in all_pairs:
        k, v = list(pair.items())[0]
        if (k in cats or v in modes) and pair not in valid:
            valid.append(pair)
    return valid

def process_csv_file(input_file, output_csv_file=None, output_json_file=None):
    try:
        df = pd.read_csv(input_file, encoding='utf-8', engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_file}")
        return

    # 1. 标准化DataFrame
    try:
        df = normalize_dataframe(df)
    except ValueError as e:
        print(f"处理文件 {input_file} 失败: {e}")
        return

    # 2. 在标准化的列上生成SFT样本
    sft_list = []
    for idx, row in df.iterrows():
        query = str(row['_query']).strip()
        if not query: continue

        history = row['_history']
        vehicle_pairs = row['_vehicle_pairs']
        
        input_str = build_input_query(history, query, vehicle_pairs)
        output_pairs = get_final_pairs(row['_gt_category'], row['_gt_mode'], vehicle_pairs)
        output_str = json.dumps(output_pairs, ensure_ascii=False)

        sft_list.append({
            "instruction": UNIFIED_INSTRUCTION,
            "input": input_str,
            "output": output_str
        })
    
    sft_list = dedup_by_input(sft_list)
    
    # 3. 文件输出
    if output_csv_file:
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        # 输出包含中间标准列的CSV，便于调试
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f'处理完成，中间过程CSV已保存到：{output_csv_file}')

    if output_json_file:
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
        with open(output_json_file, 'w', encoding='utf-8') as fout:
            json.dump(sft_list, fout, ensure_ascii=False, indent=2)
        print(f"SFT样本已保存到：{output_json_file}")


# --- 4. Batch Processing and File Management ---
def merge_json_files(file_list, output_file):
    merged_data = []
    for file_name in file_list:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): merged_data.extend(data)
                else: print(f"警告：{file_name} 不是一个列表结构，已跳过。")
        except FileNotFoundError:
            print(f"警告: 合并时未找到文件 {file_name}，可能该文件在处理时出错被跳过。")
    if merged_data:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        unique_merged_data = dedup_by_input(merged_data)
        print(f"合并前共 {len(merged_data)} 条数据，去重后剩余 {len(unique_merged_data)} 条。")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_merged_data, f, ensure_ascii=False, indent=2)
        print(f"合并完成，结果已保存到 {output_file}")
    else:
        print("没有可合并的数据，未生成合并文件。")

def batch_process(file_process_configs, merge_json_output=None):
    json_files = []
    for config in file_process_configs:
        print(f"\n--- 正在处理：{config['input_csv']} ---")
        process_csv_file(config['input_csv'], config.get('output_csv'), config.get('output_json'))
        if config.get('output_json'): json_files.append(config['output_json'])
            
    if merge_json_output and json_files:
        print("\n--- 开始合并所有JSON文件 ---")
        merge_json_files(json_files, merge_json_output)

def get_today_str():
    return time.strftime("%Y%m%d", time.localtime())

VERSION = "v6.0_Universal_Fusion"

def add_time_ver_dir(filepath):
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    new_dir = os.path.join(base_dir, get_today_str(), VERSION)
    return os.path.join(new_dir, filename)


if __name__ == "__main__":
    base_input_path = '/home/workspace/lgq/data'
    base_output_path = '/home/workspace/lgq/data/output'

    # # --- 创建演示文件 ---
    # if not os.path.exists('format1_old_json.csv'):
    #     print("创建演示文件: format1_old_json.csv")
    #     with open('format1_old_json.csv', 'w', encoding='utf-8') as f:
    #         f.write('车型理解,车列表,模型2标准答案\n"打个快车","{\\"appName\\": \\"滴滴\\",\\"taxiSkuList\\": [{\\"index\\": \\"3\\", \\"category\\": \\"推荐\\", \\"carModel\\": \\"滴滴快车\\", \\"price\\": 13},{\\"index\\": \\"2\\", \\"category\\": \\"推荐\\", \\"carModel\\": \\"特惠快车\\", \\"price\\": 12}]}","[\'3\']"')
    
    # if not os.path.exists('format2_new_multi.csv'):
    #     print("创建演示文件: format2_new_multi.csv")
    #     with open('format2_new_multi.csv', 'w', encoding='utf-8') as f:
    #         f.write('query1,query2,query3,车辆候选列表,app,大类,小类,标注人\n"打车到虹桥机场2号航站楼","用滴滴帮我叫辆专车。","打车到深圳宝安国际机场T3","|序号id|应用名称|车型大类|车辆类型|价格|\\n|---|---|---|---|---|\\n|17|滴滴|舒适|滴滴专车|52.80|\\n|18|滴滴|六座|六座专车|61.86|",,舒适,滴滴专车,王笑')

    # if not os.path.exists('format3_new_single.csv'):
    #     print("创建演示文件: format3_new_single.csv")
    #     with open('format3_new_single.csv', 'w', encoding='utf-8') as f:
    #         f.write('query,车辆候选列表,category正确答案,mode正确答案\n"从广州塔到白云机场，约个特价拼车。","|序号id|应用名称|车型大类|车辆类型|价格|\\n|---|---|---|---|---|\\n|3|滴滴|拼车|特价拼车|17.25-18.36|\\n|4|滴滴|特价|花小猪外接版|17.02|",拼车,特价拼车')

    # --- 配置文件列表 ---
    file_names = [
        '多轮query - Sheet1.csv',
        '全新训练数据20250613.csv',
        # 'format3_new_single.csv'
    ]
    
    existing_files = [fname for fname in file_names if os.path.exists(os.path.join(base_input_path, fname))]
    if not existing_files:
        print(f"警告：在路径 '{base_input_path}' 下找不到任何指定的CSV文件。请检查路径和文件名。")

    raw_file_process_configs = [
        {'input_csv': os.path.join(base_input_path, fname),
         'output_csv': os.path.join(base_output_path, fname.replace('.csv', '_debug.csv')),
         'output_json': os.path.join(base_output_path, fname.replace('.csv', '_sft.json'))
        } for fname in existing_files
    ]
    
    file_process_configs = []
    for item in raw_file_process_configs:
        new_item = dict(item)
        new_item['output_csv'] = add_time_ver_dir(item['output_csv'])
        new_item['output_json'] = add_time_ver_dir(item['output_json'])
        file_process_configs.append(new_item)

    merge_json_output = add_time_ver_dir(os.path.join(base_output_path, 'merged_sft_data.json'))
    
    batch_process(file_process_configs, merge_json_output=merge_json_output)