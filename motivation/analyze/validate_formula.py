import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# ==========================================
# 🛠️ 配置区域 (可修改变量)
# ==========================================
IMPORTANCE_FILE = "importance_rank_report_topk_30.txt"  # 请确保这是包含Kurt/Ent的新报告
UPDATE_FILE = "lora_update_magnitudes_1.txt"    # 你的LoRA更新幅度文件
OUTPUT_IMG = "full_metric_analysis.png"         # 输出图片名

# [你要求的变量] 用于计算命中率的 Top-K 数量
# 比如你想看前 66 个是否命中，就填 66
TOP_K_VAR = 66 

# ==========================================
# 1. 解析函数
# ==========================================

def parse_importance_file(filepath):
    """
    解析 importance_rank_report.txt
    支持格式: Rank | Name | Score | Norm | Kurt | Ent | ...
    """
    print(f">>> 正在解析 {filepath}...")
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if "|" not in line or "Rank" in line or "---" in line:
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        
        try:
            name = parts[1]
            if name.startswith("..."): name = name[3:]
            
            # 读取各项指标
            entry = {
                'score': float(parts[2]),
                'norm': float(parts[3]),
                # 尝试读取 Kurt 和 Ent (如果之前生成的报告里有的话)
                'kurt': float(parts[4]) if len(parts) > 4 else 0.0,
                'ent':  float(parts[5]) if len(parts) > 5 else 0.0
            }
            data[name] = entry
        except ValueError:
            continue
            
    return data

def parse_update_file(filepath):
    """
    智能解析 lora_update_magnitudes.txt (表格模式)
    自动拼接 layers.{i}.{struct}.{mod}
    """
    print(f">>> 正在解析 {filepath}...")
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or "Layer" in line or "---" in line: continue
        
        parts = line.split('|')
        if len(parts) < 3: continue
        
        try:
            layer_idx = parts[0].strip()
            mod_suffix = parts[1].strip()
            val = float(parts[2].strip())
            
            # 拼接名字
            mid = "mlp" if any(x in mod_suffix for x in ['gate','up','down']) else "self_attn"
            full_name = f"layers.{layer_idx}.{mid}.{mod_suffix}"
            data[full_name] = val
        except: continue
        
    return data

# ==========================================
# 2. 核心分析逻辑
# ==========================================

def analyze_and_plot(imp_data, upd_data, top_k_cutoff):
    # 1. 数据对齐
    common = set(imp_data.keys()) & set(upd_data.keys())
    print(f">>> 找到 {len(common)} 个匹配模块进行分析。")
    if len(common) < 10:
        print("❌ 匹配模块过少，请检查文件格式。")
        return

    # 2. 提取数据列表
    names = []
    vec_update = []
    vec_score = []
    vec_norm = []
    vec_kurt = []
    vec_ent = []

    for name in common:
        names.append(name)
        vec_update.append(upd_data[name])
        vec_score.append(imp_data[name]['score'])
        vec_norm.append(imp_data[name]['norm'])
        vec_kurt.append(imp_data[name]['kurt'])
        vec_ent.append(imp_data[name]['ent'])

    # 转 numpy
    v_update = np.array(vec_update)
    
    # 定义要分析的指标列表
    metrics = [
        ("Score (My Formula)", np.array(vec_score), "red", "Negative"),
        ("Norm", np.array(vec_norm), "green", "Positive"),
        ("Kurtosis", np.array(vec_kurt), "blue", "Unknown"),
        ("Entropy", np.array(vec_ent), "purple", "Unknown")
    ]

    # 3. 打印相关性报告 & 绘图
    plt.figure(figsize=(20, 5)) # 宽图，放4个子图
    
    print("\n" + "="*60)
    print("📊 全指标相关性分析报告")
    print("="*60)

    for i, (label, v_data, color, expect) in enumerate(metrics):
        # 计算相关性
        p_corr, _ = pearsonr(v_data, v_update) # 线性
        s_corr, _ = spearmanr(v_data, v_update) # 排名
        
        print(f"{i+1}. {label} vs Update:")
        print(f"   - Pearson:  {p_corr:.4f}")
        print(f"   - Spearman: {s_corr:.4f}")
        print(f"   - 预期方向: {expect}")
        if abs(s_corr) > 0.5: print(f"   🔥 强相关! (这是关键指标)")
        print("-" * 30)

        # 绘图
        plt.subplot(1, 4, i+1)
        sns.regplot(x=v_data, y=v_update, scatter_kws={'alpha':0.4}, line_kws={'color':color})
        plt.title(f"{label}\nCorr: {s_corr:.2f}")
        plt.xlabel(label)
        plt.ylabel("LoRA Update Magnitude")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\n>>> 分析图表已保存至: {OUTPUT_IMG}")

    # 4. Top-K 命中率分析 (Recall Analysis)
    print(f"\n📈 Top-{top_k_cutoff} 关键层命中率 (Recall Analysis)")
    print("-" * 60)
    
    # Ground Truth: 实际更新最大的 Top-K
    # update 越大越重要 -> reverse=True
    top_k_actual = set([x for _, x in sorted(zip(v_update, names), reverse=True)][:top_k_cutoff])
    
    print(f"基准: 实际更新幅度最大的 {top_k_cutoff} 层")
    
    # 策略 1: 你的公式 (Score 越低越重要)
    top_k_score = set([x for _, x in sorted(zip(vec_score, names))][:top_k_cutoff])
    recall_score = len(top_k_score & top_k_actual) / top_k_cutoff
    print(f"👉 [公式策略] 命中率: {recall_score:.2%} ({len(top_k_score & top_k_actual)}/{top_k_cutoff})")

    # 策略 2: 纯 Norm (Norm 越大越重要)
    top_k_norm = set([x for _, x in sorted(zip(vec_norm, names), reverse=True)][:top_k_cutoff])
    recall_norm = len(top_k_norm & top_k_actual) / top_k_cutoff
    print(f"👉 [Norm策略] 命中率: {recall_norm:.2%} ({len(top_k_norm & top_k_actual)}/{top_k_cutoff})")

    # 策略 3: 纯 Kurtosis (假设 Norm 是对的，Kurt 一般也是正相关或负相关，我们试一下反向)
    # 我们可以根据上面的 Corr 正负自动决定方向
    k_corr = spearmanr(vec_kurt, v_update)[0]
    reverse_k = True if k_corr > 0 else False
    dir_str = "高->低" if reverse_k else "低->高"
    
    top_k_kurt = set([x for _, x in sorted(zip(vec_kurt, names), reverse=reverse_k)][:top_k_cutoff])
    recall_kurt = len(top_k_kurt & top_k_actual) / top_k_cutoff
    print(f"👉 [Kurt策略] 命中率: {recall_kurt:.2%} (方向: {dir_str})")

# ==========================================
# 🚀 主程序
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(IMPORTANCE_FILE) or not os.path.exists(UPDATE_FILE):
        print("❌ 错误: 找不到输入文件。")
        print(f"请确保 {IMPORTANCE_FILE} 和 {UPDATE_FILE} 都在当前目录下。")
        exit()
        
    # 解析数据
    imp_data = parse_importance_file(IMPORTANCE_FILE)
    upd_data = parse_update_file(UPDATE_FILE)
    
    # 运行全量分析
    analyze_and_plot(imp_data, upd_data, TOP_K_VAR)