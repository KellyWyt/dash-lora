import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import os

# ==========================================
# 🛠️ 配置区域 (请在这里修改参数)
# ==========================================

# 1. 模型路径 (请修改为你本地 LLaMA 权重的实际路径)
MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf" 

# 2. 独享层的总预算 (Top-K)
# 建议值: 总层数(32) * Proj类型(7) * 30% ≈ 64
TOP_K_BUDGET = 56

# 3. 输出报告的文件名
OUTPUT_FILE = "importance_rank_report_topk_56.txt"

# 4. 需要分析的目标模块名称
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# ==========================================
# 1. 基础数学计算函数
# ==========================================

def compute_kurtosis(tensor_data):
    """计算峰度：峰度越高 -> 越稳定"""
    if tensor_data.std() < 1e-8: return 0
    k = scipy_kurtosis(tensor_data.flatten(), fisher=True)
    return 1 / (1 + np.exp(-k / 5))

def compute_entropy(tensor_data, bins=50):
    """计算熵：熵越高 -> 越稳定 (基于之前的假设)"""
    hist, _ = np.histogram(tensor_data.flatten(), bins=bins, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    ent = scipy_entropy(hist)
    return ent / np.log(bins)

def compute_norm(tensor_data):
    """计算范数：范数越大 -> 越不稳定"""
    return torch.norm(torch.tensor(tensor_data).float()).item()

def minmax_normalize(arr):
    """归一化辅助函数"""
    _min, _max = arr.min(), arr.max()
    if _max - _min < 1e-6: return np.zeros_like(arr) + 0.5
    return (arr - _min) / (_max - _min)

# ==========================================
# 2. 核心分析逻辑
# ==========================================

def get_raw_metrics(model, target_modules_list):
    """计算原始指标"""
    print(f">>> 正在扫描模型层 (目标: {target_modules_list})...")
    raw_metrics = {}
    valid_modules = []
    
    for name, module in model.named_modules():
        if any(t in name for t in target_modules_list) and isinstance(module, nn.Linear):
            valid_modules.append((name, module))
    
    print(f">>> 找到 {len(valid_modules)} 个模块，开始计算 (CPU)...")

    for name, module in tqdm(valid_modules):
        # 转换为 float32 numpy 数组
        w = module.weight.data.float().numpy()
        raw_metrics[name] = {
            'kurtosis': compute_kurtosis(w),
            'entropy': compute_entropy(w),
            'norm': compute_norm(w)
        }
    return raw_metrics

def calculate_final_scores(raw_metrics, target_modules_list):
        # """
        #         === 修改点 1: 纯 Norm 打分公式 ===
        #         目标: Norm 越大 -> 更新幅度越大 -> 越不稳定 -> Score 应越低 (排前面)
        #         """
        grouped = {}
        # 自动识别类型
        all_types = set()
        for name in raw_metrics:
            for pt in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
                if pt in name: all_types.add(pt); break
        
        for pt in all_types:
            grouped[pt] = {'k': [], 'e': [], 'n': [], 'names': []}

        for name, m in raw_metrics.items():
            for pt in all_types:
                if pt in name:
                    grouped[pt]['k'].append(m['kurtosis'])
                    grouped[pt]['e'].append(m['entropy'])
                    grouped[pt]['n'].append(m['norm'])
                    grouped[pt]['names'].append(name)
                    break
        
        final_scores = {}
        
        for pt, data in grouped.items():
            if not data['names']: continue
            
            # 虽然我们只用 Norm，但为了归一化，还是需要提取出来
            n = np.array(data['n'])
            
            # Min-Max 归一化
            def minmax(arr):
                _min, _max = arr.min(), arr.max()
                if _max - _min < 1e-6: return np.zeros_like(arr) + 0.5
                return (arr - _min) / (_max - _min)

            norm_n = minmax(n)
            
            # === 核心公式修改 ===
            # 只考虑 Norm。Norm 越大(norm_n接近1)，Score越小(接近0)。
            # 这样在升序排序时，Norm 大的层会排在最前面 (Rank 1, 2, 3...)
            scores = 1.0 - norm_n
            
            for i, name in enumerate(data['names']):
                final_scores[name] = scores[i]
                
        return final_scores

def calculate_final_scores1(raw_metrics, target_modules_list):
    """计算最终分数"""
    print(">>> 正在计算归一化分数...")
    final_scores = {}
    
    # 按投影类型分组 (q_proj, v_proj...) 进行归一化
    grouped_data = {}
    for name in raw_metrics:
        for pt in target_modules_list:
            if pt in name:
                if pt not in grouped_data:
                    grouped_data[pt] = {'k': [], 'e': [], 'n': [], 'names': []}
                grouped_data[pt]['k'].append(raw_metrics[name]['kurtosis'])
                grouped_data[pt]['e'].append(raw_metrics[name]['entropy'])
                grouped_data[pt]['n'].append(raw_metrics[name]['norm'])
                grouped_data[pt]['names'].append(name)
                break
    
    for pt, data in grouped_data.items():
        if not data['names']: continue
        
        # 组内归一化
        norm_k = minmax_normalize(np.array(data['k']))
        norm_e = minmax_normalize(np.array(data['e']))
        norm_n = minmax_normalize(np.array(data['n']))
        
        # [公式] Score越低 = 越不稳定 = 排名越前(Exclusive)
        # Norm 越大 -> (1-Norm) 越小 -> Score 越低
        scores = 0.3 * norm_k + 0.3 * norm_e + 0.4 * (1.0 - norm_n)
        
        for i, name in enumerate(data['names']):
            final_scores[name] = scores[i]
            
    return final_scores

def determine_allocation(final_scores, budget):
    """非对称分配策略"""
    print(f">>> 正在执行非对称分配 (预算: {budget})...")
    
    # 排序：分数从低到高 (Most Unstable First)
    sorted_items = sorted(final_scores.items(), key=lambda x: x[1])
    all_modules = [x[0] for x in sorted_items]
    
    # 70% 给 Text, 30% 给 Others
    limit_text = int(budget * 1)
    limit_others = int(budget * 0.5)
    
    # 边界检查
    limit_text = min(max(1, limit_text), len(all_modules))
    limit_others = min(max(0, limit_others), len(all_modules))
    
    text_exclusive = set(all_modules[:limit_text])
    others_exclusive = set(all_modules[:limit_others])
    
    allocations = {}
    for i, name in enumerate(all_modules):
        allocations[name] = {
            'rank': i + 1,
            'score': final_scores[name],
            'text_mode': 'EXCLUSIVE' if name in text_exclusive else 'SHARED',
            'others_mode': 'EXCLUSIVE' if name in others_exclusive else 'SHARED'
        }
    return allocations, limit_text, limit_others

def save_report(filename, allocations, raw_metrics, l_text, l_others):
    print(f">>> 正在保存详细结果到: {filename}")
    sorted_names = sorted(allocations.keys(), key=lambda k: allocations[k]['rank'])
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*140 + "\n")
        f.write(f" MODEL IMPORTANCE ANALYSIS REPORT (FULL METRICS)\n")
        f.write("="*140 + "\n")
        f.write(f"Model Path:    {MODEL_PATH}\n")
        f.write(f"Total Budget:  {TOP_K_BUDGET}\n")
        f.write("-" * 140 + "\n")
        # 增加了 Kurt 和 Ent 列
        header = f"{'Rank':<6} | {'Module Name':<50} | {'Score':<8} | {'Norm':<8} | {'Kurt':<8} | {'Ent':<8} | {'Text':<10} | {'Others':<10}\n"
        f.write(header)
        f.write("-" * 140 + "\n")
        
        for name in sorted_names:
            info = allocations[name]
            raw = raw_metrics[name]
            disp_name = "..." + name[-45:] if len(name) > 48 else name
            
            # 写入所有指标
            row = f"{info['rank']:<6} | {disp_name:<50} | {info['score']:.4f}   | {raw['norm']:<8.2f} | {raw['kurtosis']:<8.4f} | {raw['entropy']:<8.4f} | {info['text_mode']:<10} | {info['others_mode']:<10}\n"
            f.write(row)
            
    print(">>> 报告保存完成（包含 Kurt/Ent 数据）。")

# ==========================================
# 🚀 主程序
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型路径: {MODEL_PATH}")
        exit(1)

    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 使用 float16 加载节省内存，计算时会自动转 float32
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        config=config, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        trust_remote_code=True
    )

    # 2. 运行流程
    raw_data = get_raw_metrics(model, TARGET_MODULES)
    scores = calculate_final_scores(raw_data, TARGET_MODULES)
    alloc_result, limit_t, limit_o = determine_allocation(scores, TOP_K_BUDGET)
    
    # 3. 保存
    save_report(OUTPUT_FILE, alloc_result, raw_data, limit_t, limit_o)