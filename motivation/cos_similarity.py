import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from scipy.stats import entropy

# ================= ⚙️ 配置路径 (请修改) =================
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/motivation_similarity_analysis"

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TARGET_LAYERS = range(32)

# ================= 1. 基础指标计算 (SIS Score) =================
# 复用之前的逻辑，计算 Kurtosis 和 Entropy
def compute_base_metrics(weight_chunk, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    results = {}
    for name, weight in weight_chunk:
        try:
            w = weight.to(device, dtype=torch.float32)
            # Kurtosis
            if w.std() < 1e-8:
                kurt = 0.0
            else:
                mean, std = w.mean(), w.std()
                kurt = ((w - mean) ** 4).mean() / (std ** 4) - 3
                kurt = kurt.item()
            
            # Entropy
            bins = 100
            min_val, max_val = w.min(), w.max()
            if (max_val - min_val) < 1e-8:
                ent_norm = 0.0
            else:
                hist = torch.histc(w, bins=bins, min=min_val, max=max_val)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                ent = -torch.sum(hist * torch.log(hist))
                max_ent = torch.log(torch.tensor(bins, device=device))
                ent_norm = (ent / max_ent).item()
            
            results[name] = {'kurt': kurt, 'ent': ent_norm}
            del w
        except Exception:
            results[name] = {'kurt': 0.0, 'ent': 1.0}
    torch.cuda.empty_cache()
    return results

# ================= 2. 相似度计算核心 =================
def compute_similarity(adapter_state):
    """
    计算每一层 Video (A1) 和 Audio (A2) 的余弦相似度
    """
    sim_data = []
    print(">>> Calculating Cross-Modal Cosine Similarity (Video vs Audio)...")
    
    for i in TARGET_LAYERS:
        for proj in TARGET_MODULES:
            # 构造 Key
            # 假设结构: layers.0.self_attn.q_proj.lora_A1.weight
            # 需要根据你的实际 bin 文件 key 格式调整，这里使用模糊匹配
            prefix = f"layers.{i}."
            proj_str = f".{proj}."
            
            # 找到该层该模块所有的 Key
            candidates = [k for k in adapter_state.keys() if prefix in k and proj_str in k]
            
            key_A1 = next((k for k in candidates if "lora_A1" in k), None) # Video
            key_A2 = next((k for k in candidates if "lora_A2" in k), None) # Audio
            
            if key_A1 and key_A2:
                # 提取参数并拉平
                A1_vec = adapter_state[key_A1].float().flatten()
                A2_vec = adapter_state[key_A2].float().flatten()
                
                # 计算余弦相似度 (-1 ~ 1)
                sim = F.cosine_similarity(A1_vec.unsqueeze(0), A2_vec.unsqueeze(0)).item()
                
                sim_data.append({
                    'layer_idx': i,
                    'module_type': proj,
                    'similarity': sim
                })
            else:
                pass 
                # print(f"Missing A1 or A2 for Layer {i} {proj}")
                
    return pd.DataFrame(sim_data)

# ================= 3. 主程序 =================
def main():
    mp.set_start_method('spawn', force=True)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step A: 计算/获取 SIS Score (X轴) ---
    print(f"Loading Base Model: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    
    tasks = []
    for name, module in model.named_modules():
        for i in TARGET_LAYERS:
            if f"layers.{i}." in name:
                for t in TARGET_MODULES:
                    if name.endswith(t) and hasattr(module, 'weight'):
                        tasks.append((name, module.weight.data.clone()))
                break
    del model

    print("Calculating Base Model Stats...")
    available_gpus = [i for i in range(torch.cuda.device_count())]
    num_gpus = len(available_gpus) if available_gpus else 1
    chunks = [tasks[i:i + len(tasks)//num_gpus + 1] for i in range(0, len(tasks), len(tasks)//num_gpus + 1)]
    
    raw_metrics = {}
    with mp.Pool(num_gpus) as pool:
        async_results = [pool.apply_async(compute_base_metrics, (c, available_gpus[i%num_gpus])) for i, c in enumerate(chunks)]
        for res in async_results: raw_metrics.update(res.get())

    # --- Step B: 计算 Similarity (Y轴) ---
    print(f"Loading Adapter: {ADAPTER_BIN_PATH}")
    adapter = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    df_sim = compute_similarity(adapter)
    
    # --- Step C: 合并数据 ---
    final_records = []
    for idx, row in df_sim.iterrows():
        layer = row['layer_idx']
        mod = row['module_type']
        
        # 找对应的 metric
        # 这里的 name 匹配比较 tricky，需要遍历 raw_metrics
        # 简单起见，我们重新构建匹配
        metric_key = None
        for k in raw_metrics.keys():
            if f"layers.{layer}." in k and k.endswith(f".{mod}"):
                metric_key = k
                break
        
        if metric_key:
            stats = raw_metrics[metric_key]
            final_records.append({
                'layer_idx': layer,
                'module_type': mod,
                'similarity': row['similarity'],
                'kurtosis': stats['kurt'],
                'entropy': stats['ent']
            })
            
    df = pd.DataFrame(final_records)
    
    # --- Step D: 归一化 & 计算 Score ---
    # Type-Aware Normalization
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-6)
    
    df['norm_kurt'] = df.groupby('module_type')['kurtosis'].transform(norm)
    df['norm_ent'] = df.groupby('module_type')['entropy'].transform(norm)
    
    # SIS Score Formula
    df['sis_score'] = 0.6 * df['norm_kurt'] + 0.4 * (1 - df['norm_ent'])
    
    # 保存数据
    df.to_csv(os.path.join(OUTPUT_DIR, "similarity_analysis_data.csv"), index=False)
    
    # ================= 4. 画图 (可视化证明) =================
    plot_analysis(df)

def plot_analysis(df):
    sns.set_theme(style="whitegrid")
    
    # --- 图 1: Layer-wise Similarity Trend (层级分析) ---
    # 目的：展示深层是否更相似？
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x="layer_idx", y="similarity", hue="module_type", marker="o", linewidth=2)
    plt.title("Analysis 1: Cross-Modal Similarity vs. Layer Depth\n(Are deep layers more shareable?)", fontsize=14)
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity (Video vs Audio)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_layer_similarity_trend.png"))
    print("Saved Plot 1")

    # --- 图 2: Module-wise Similarity Distribution (模块分析) ---
    # 目的：展示哪些模块天生相似度高？
    plt.figure(figsize=(12, 6))
    order = df.groupby('module_type')['similarity'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="module_type", y="similarity", order=order, palette="viridis")
    plt.title("Analysis 2: Shareability by Module Type\n(Higher = More Shareable)", fontsize=14)
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_module_similarity_dist.png"))
    print("Saved Plot 2")

    # --- 图 3: Motivation Proof (Correlation) ---
    # 目的：High SIS Score -> High Similarity?
    plt.figure(figsize=(12, 8))
    
    # 计算相关系数
    corrs = []
    for mod in df['module_type'].unique():
        sub = df[df['module_type'] == mod]
        r = sub['sis_score'].corr(sub['similarity'])
        corrs.append(f"{mod}: r={r:.2f}")
        # 画散点和回归线
        sns.regplot(data=sub, x="sis_score", y="similarity", label=f"{mod} (r={r:.2f})", scatter_kws={'alpha':0.6}, ci=None)
    
    plt.title("Analysis 3: SIS Score vs. Cross-Modal Similarity\n(Hypothesis: Positive Correlation -> High Score means Shareable)", fontsize=14)
    plt.xlabel("SIS Score (Pre-trained Stability)\nHigh Score = Stable")
    plt.ylabel("Cosine Similarity (Video vs Audio)\nHigh Similarity = Shareable")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_motivation_proof_similarity.png"))
    print("Saved Plot 3")
    
    print("\n>>> Analysis Complete!")
    print("Check the plots in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()