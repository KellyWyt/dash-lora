import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
import os
from tqdm import tqdm

# ================= 配置路径 =================
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"

# 输出文件路径
SCORE_FILE = "score_formula_and_values_1.txt"
LORA_MAGNITUDE_FILE = "lora_update_magnitudes_1.txt"

# 目标模块
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# ================= 1. 统计计算函数 =================
def compute_kurtosis(tensor):
    tensor = tensor.float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0: return 0.0
    return torch.mean((tensor - mean)**4) / (std**4) - 3.0

def compute_entropy(tensor, bins=100):
    tensor = tensor.float().cpu().numpy().flatten()
    hist, _ = np.histogram(tensor, bins=bins, density=True)
    hist = hist + 1e-12
    return entropy(hist)

def compute_sparsity(tensor, threshold=1e-4):
    return (torch.abs(tensor) < threshold).float().mean().item()

def compute_effective_rank(tensor):
    if tensor.dim() < 2: return 0.5
    try:
        # SVD 在 GPU 上快很多
        if torch.cuda.is_available():
            tensor = tensor.to("cuda")
        S = torch.linalg.svdvals(tensor.float())
        p = S / torch.sum(S)
        p = p[p > 0]
        entropy_val = -torch.sum(p * torch.log(p))
        return torch.exp(entropy_val).item()
    except Exception as e:
        return 0.0

# ================= 2. 数据收集主逻辑 =================
def main():
    print(f"Loading Adapter from {ADAPTER_BIN_PATH} ...")
    adapter_state = torch.load(ADAPTER_BIN_PATH, map_location="cpu")
    
    print(f"Loading Base Model from {BASE_MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    num_layers = len(model.model.layers)
    data_records = []

    print(">>> Start scanning layers and calculating metrics...")
    
    for i in tqdm(range(num_layers), desc="Analyzing Layers"):
        layer = model.model.layers[i]
        
        # 映射模块名到对象
        modules_map = {
            'q_proj': layer.self_attn.q_proj,
            'k_proj': layer.self_attn.k_proj,
            'v_proj': layer.self_attn.v_proj,
            'o_proj': layer.self_attn.o_proj,
            'gate_proj': layer.mlp.gate_proj,
            'up_proj': layer.mlp.up_proj,
            'down_proj': layer.mlp.down_proj
        }

        for m_name, module in modules_map.items():
            # --- A. Base Model Stats ---
            weight = module.weight.data
            # 放到 GPU 计算指标
            if torch.cuda.is_available():
                weight_gpu = weight.to("cuda")
            else:
                weight_gpu = weight
            
            kurt = compute_kurtosis(weight_gpu).item()
            ent = compute_entropy(weight_gpu)
            spar = compute_sparsity(weight_gpu)
            rank = compute_effective_rank(weight_gpu)
            
            # --- B. Ground Truth (LoRA Norm) ---
            # 查找该层该模块对应的 adapter
            layer_prefix = f"layers.{i}."
            proj_pattern = f".{m_name}."
            
            # 模糊匹配 keys
            keys_A = [k for k in adapter_state.keys() if layer_prefix in k and proj_pattern in k and "lora_A" in k]
            
            gt_norm = 0.0
            found_adapter = False
            for kA in keys_A:
                # 寻找对应的 B
                # 尝试1: 同名 B
                kB = kA.replace("lora_A", "lora_B")
                # 尝试2: 共享 B (default)
                if kB not in adapter_state:
                    kB_shared = kA.split("lora_A")[0] + "lora_B.default.weight"
                    if kB_shared in adapter_state:
                        kB = kB_shared
                
                if kB in adapter_state:
                    found_adapter = True
                    A = adapter_state[kA].float()
                    B = adapter_state[kB].float()
                    # Norm Calculation
                    delta = torch.norm(torch.matmul(B, A)).item()
                    gt_norm += delta
            
            if not found_adapter:
                # 如果这一层没有被 LoRA 训练（比如被冻结了），Norm 为 0
                gt_norm = 0.0

            data_records.append({
                'layer_idx': i,
                'module_type': m_name,
                'kurtosis': kurt,
                'entropy': ent,
                'sparsity': spar,
                'effective_rank': rank,
                'lora_norm_gt': gt_norm
            })

    # ================= 3. 数据处理与分组归一化 (Pandas) =================
    df = pd.DataFrame(data_records)
    
    print("\n>>> Performing Type-Aware Normalization (Group by Module Type)...")
    
    # 定义归一化函数
    def min_max_norm(x):
        if x.max() == x.min(): return x # 防止除以0
        return (x - x.min()) / (x.max() - x.min())

    # 对每一列指标，在 module_type 内部进行归一化
    # 这样 gate_proj 的 rank 不会和 v_proj 的 rank 混淆
    df['norm_kurt'] = df.groupby('module_type')['kurtosis'].transform(min_max_norm)
    df['norm_ent']  = df.groupby('module_type')['entropy'].transform(min_max_norm)
    df['norm_spar'] = df.groupby('module_type')['sparsity'].transform(min_max_norm)
    df['norm_rank'] = df.groupby('module_type')['effective_rank'].transform(min_max_norm)

    # 计算最终分数 (按照你的公式)
    # High Score = High Stability = Low Fine-tuning Need
    # Entropy 越低越有序，所以用 (1-Entropy)
    df['final_score'] = (
        0.3 * df['norm_kurt'] + 
        0.3 * (1 - df['norm_ent']) + 
        0.2 * df['norm_spar'] + 
        0.2 * df['norm_rank']
    )

    # ================= 4. 保存文件 =================
    
    # --- 文件 1: 公式与分数 ---
    with open(SCORE_FILE, "w") as f:
        f.write("=== Algorithm: Statistic-based Importance Scoring (SIS) ===\n")
        f.write("Formula: Score = 0.3*Norm(Kurt) + 0.3*(1-Norm(Ent)) + 0.2*Norm(Spar) + 0.2*Norm(Rank)\n")
        f.write("Note: Normalization is performed WITHIN each module type (Type-Aware).\n")
        f.write("Hypothesis: Higher Score indicates higher pre-trained stability, thus LOWER fine-tuning need.\n\n")
        f.write(f"{'Layer':<6} | {'Module':<10} | {'Kurt':<8} | {'Ent':<8} | {'Spar':<8} | {'Rank':<8} || {'FINAL_SCORE':<12}\n")
        f.write("-" * 80 + "\n")
        
        # 按分数排序（可选，这里按层数打印更直观）
        for _, row in df.iterrows():
            f.write(f"{int(row['layer_idx']):<6} | {row['module_type']:<10} | "
                    f"{row['kurtosis']:.2f}     | {row['entropy']:.2f}     | "
                    f"{row['sparsity']:.4f}   | {row['effective_rank']:.1f}      || "
                    f"{row['final_score']:.6f}\n")
    
    print(f"Saved scores to: {SCORE_FILE}")

    # --- 文件 2: LoRA 微调强度 (Ground Truth) ---
    with open(LORA_MAGNITUDE_FILE, "w") as f:
        f.write("=== Ground Truth: Actual LoRA Update Magnitudes (Frobenius Norm) ===\n")
        f.write("Used for correlation verification.\n\n")
        f.write(f"{'Layer':<6} | {'Module':<10} | {'LoRA_Update_Norm':<15}\n")
        f.write("-" * 40 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{int(row['layer_idx']):<6} | {row['module_type']:<10} | {row['lora_norm_gt']:.6f}\n")

    print(f"Saved magnitudes to: {LORA_MAGNITUDE_FILE}")

    # ================= 5. 验证动机：分组相关性分析 =================
    print("\n======== Correlation Analysis (Per Module Type) ========")
    print("Hypothesis: Correlation should be NEGATIVE (High Score -> Low Update)")
    
    avg_corr = 0
    valid_modules = 0
    
    # 准备画图数据
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_MODULES)))

    for idx, m_type in enumerate(TARGET_MODULES):
        sub_df = df[df['module_type'] == m_type]
        if len(sub_df) == 0: continue
        
        # 计算相关性
        corr = sub_df['final_score'].corr(sub_df['lora_norm_gt'])
        print(f"Module: {m_type:<10} | Correlation: {corr:.4f}")
        
        if not np.isnan(corr):
            avg_corr += corr
            valid_modules += 1

        # 画散点
        plt.scatter(
            sub_df['final_score'], 
            sub_df['lora_norm_gt'], 
            label=f"{m_type} (r={corr:.2f})",
            color=colors[idx],
            alpha=0.7
        )
        
        # 画同颜色的趋势线
        z = np.polyfit(sub_df['final_score'], sub_df['lora_norm_gt'], 1)
        p = np.poly1d(z)
        plt.plot(sub_df['final_score'], p(sub_df['final_score']), color=colors[idx], linestyle='--')

    print("-" * 40)
    if valid_modules > 0:
        print(f"Average Correlation: {avg_corr / valid_modules:.4f}")

    plt.title("Type-Aware Motivation Proof: Normalized Score vs. LoRA Update")
    plt.xlabel("Proposed Score (Type-Aware Normalized)\n(High = Stable)")
    plt.ylabel("Actual LoRA Update Norm")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("motivation_improved_proof.png", dpi=300)
    print("\nPlot saved to motivation_improved_proof_1.png")

    # ================= 6. 辅助分析：哪个单一指标最强？ =================
    print("\n======== Metric Ablation Study (Which metric is best?) ========")
    metrics = ['norm_kurt', 'norm_ent', 'norm_spar', 'norm_rank']
    best_corr = 0
    best_metric = ""
    
    for met in metrics:
        # 计算所有模块内的平均相关性
        met_corrs = []
        for m_type in TARGET_MODULES:
            sub_df = df[df['module_type'] == m_type]
            # 注意 Entropy 原义是反向的，我们在公式里取了反。
            # 这里直接看归一化后的指标与 Norm 的关系
            if met == 'norm_ent': 
                # 这里 norm_ent 是归一化的熵 (越大约乱) -> 期望正相关
                c = sub_df[met].corr(sub_df['lora_norm_gt'])
            else:
                # 其他指标 -> 期望负相关
                c = sub_df[met].corr(sub_df['lora_norm_gt'])
            if not np.isnan(c): met_corrs.append(c)
            
        avg = np.mean(met_corrs) if met_corrs else 0
        print(f"Metric: {met:<10} | Avg Corr: {avg:.4f}")

if __name__ == "__main__":
    main()