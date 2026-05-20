import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
from tqdm import tqdm
import re

# ================= 配置路径 =================
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"
# ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"

# 目标模块类型
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# ================= 1. 统计指标计算函数 (Your Formula) =================
def compute_kurtosis(tensor):
    # Fisher kurtosis (normal ==> 0.0)
    tensor = tensor.float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0: return 0.0
    return torch.mean((tensor - mean)**4) / (std**4) - 3.0

def compute_entropy(tensor, bins=100):
    # 使用直方图计算分布熵
    tensor = tensor.float().cpu().numpy().flatten()
    hist, _ = np.histogram(tensor, bins=bins, density=True)
    # 加上微小值防止 log(0)
    hist = hist + 1e-12
    return entropy(hist)

def compute_sparsity(tensor, threshold=1e-4):
    # 软稀疏度：绝对值小于阈值的比例
    return (torch.abs(tensor) < threshold).float().mean().item()

def compute_effective_rank(tensor):
    # 有效秩：奇异值的 Shannon Entropy
    # 注意：SVD 非常耗时，对于 4096*4096 的矩阵，建议在 GPU 上做
    if tensor.dim() < 2: return 0.5
    try:
        # 转换为 float32 防止精度溢出
        # 只需要奇异值 (S)，不需要 U 和 V
        S = torch.linalg.svdvals(tensor.float())
        # 归一化奇异值分布
        p = S / torch.sum(S)
        # 计算熵
        p = p[p > 0] # 去除 0
        entropy_val = -torch.sum(p * torch.log(p))
        return torch.exp(entropy_val).item()
    except Exception as e:
        print(f"SVD Error: {e}")
        return 0.0

# ================= 2. 主逻辑 =================

def main():
    print(f"Loading Adapter weights from {ADAPTER_BIN_PATH} ...")
    adapter_state = torch.load(ADAPTER_BIN_PATH, map_location="cpu")
    
    print(f"Loading Base Model (on CPU/Disk to save GPU RAM for SVD) from {BASE_MODEL_PATH} ...")
    # 使用 device_map="auto" 或者 "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    num_layers = len(model.model.layers)
    print(f"Model loaded. Total Layers: {num_layers}")

    # 存储原始分数
    raw_metrics = {
        'kurtosis': [], 'entropy': [], 'sparsity': [], 'effective_rank': [],
        'layer_idx': [], 'module_type': [], 'lora_norm_gt': []
    }

    # --- 遍历每一层，计算 Base Score 和 Ground Truth ---
    print("Start analyzing layers (This may take time due to SVD)...")
    
    for i in tqdm(range(num_layers), desc="Processing Layers"):
        layer = model.model.layers[i]
        
        # 遍历该层感兴趣的模块
        # 通过 getattr 获取模块对象
        # 注意: LlamaMLP 的结构是 layer.mlp.gate_proj 等
        # LlamaAttention 的结构是 layer.self_attn.q_proj 等
        
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
            weight = module.weight.data # Base Weight
            
            # 1. 计算 Base Model 的统计指标
            # 将权重移到 GPU 计算指标，然后移回
            if torch.cuda.is_available():
                weight_gpu = weight.to("cuda")
            else:
                weight_gpu = weight
            
            kurt = compute_kurtosis(weight_gpu).item()
            ent = compute_entropy(weight_gpu)
            spar = compute_sparsity(weight_gpu)
            rank = compute_effective_rank(weight_gpu)
            
            raw_metrics['kurtosis'].append(kurt)
            raw_metrics['entropy'].append(ent)
            raw_metrics['sparsity'].append(spar)
            raw_metrics['effective_rank'].append(rank)
            raw_metrics['layer_idx'].append(i)
            raw_metrics['module_type'].append(m_name)
            
            # 2. 计算 Ground Truth (LoRA Update Norm)
            # 需要在 adapter_state 中找到对应的 A 和 B
            # Key 格式通常是: base_model.model.model.layers.{i}.{attn/mlp}.{proj}.lora_A.{adapter}.weight
            # 我们需要模糊匹配
            
            layer_prefix = f"layers.{i}."
            proj_pattern = f".{m_name}."
            
            # 找到属于该层、该模块的所有 LoRA A 和 B
            relevant_keys_A = [k for k in adapter_state.keys() if layer_prefix in k and proj_pattern in k and "lora_A" in k]
            
            total_delta_norm = 0.0
            
            for kA in relevant_keys_A:
                kB = kA.replace("lora_A", "lora_B")
                # 尝试处理共享 B 的情况 (如果 kB 不存在，找 default B)
                if kB not in adapter_state:
                    # 这是一个简单的 fallback，具体要看你的 key 结构
                    # 假设 moka 可能是 lora_B.default
                    kB_base = kA.split("lora_A")[0] + "lora_B.default.weight"
                    if kB_base in adapter_state:
                        kB = kB_base
                
                if kB in adapter_state:
                    A = adapter_state[kA].float() # 用 float32 计算更准
                    B = adapter_state[kB].float()
                    # Norm calculation
                    delta = torch.norm(torch.matmul(B, A)).item()
                    total_delta_norm += delta
            
            raw_metrics['lora_norm_gt'].append(total_delta_norm)

    # ================= 3. 归一化与最终分数计算 =================
    print("Computing Final Scores with Normalization...")
    
    # 辅助归一化函数
    def normalize_list(values):
        v = np.array(values)
        return (v - v.min()) / (v.max() - v.min() + 1e-8)

    norm_kurt = normalize_list(raw_metrics['kurtosis'])
    norm_ent  = normalize_list(raw_metrics['entropy'])
    norm_spar = normalize_list(raw_metrics['sparsity'])
    norm_rank = normalize_list(raw_metrics['effective_rank'])
    
    # 你的公式
    # Score = 0.3*Kurt + 0.3*(1-Entropy) + 0.2*Sparsity + 0.2*Rank
    # 注意：Entropy 越小越有序，(1-Ent) 越大代表越有序/稳定
    final_scores = (
        0.3 * norm_kurt + 
        0.3 * (1 - norm_ent) + 
        0.2 * norm_spar + 
        0.2 * norm_rank
    )
    
    gt_norms = np.array(raw_metrics['lora_norm_gt'])

    # ================= 4. 验证动机：计算相关性与画图 =================
    
    # 计算 Pearson 相关系数
    correlation = np.corrcoef(final_scores, gt_norms)[0, 1]
    print(f"\n======== 实验结果 ========")
    print(f"Correlation between Base_Score and LoRA_Update: {correlation:.4f}")
    if correlation < 0:
        print(">> 成功！呈现负相关。说明分数越高（权重越稳），微调更新越小。")
        print(">> 结论：应给分数低的层分配更多参数。")
    else:
        print(">> 警告：呈现正相关或无关。可能需要调整公式权重的符号。")

    # 画散点图
    plt.figure(figsize=(10, 6))
    
    # 为了看清不同模块，可以用颜色区分
    unique_modules = list(set(raw_metrics['module_type']))
    colors = plt.cm.get_cmap('tab10', len(unique_modules))
    
    for idx, mod_name in enumerate(unique_modules):
        indices = [i for i, x in enumerate(raw_metrics['module_type']) if x == mod_name]
        plt.scatter(
            final_scores[indices], 
            gt_norms[indices], 
            alpha=0.6, 
            label=mod_name,
            s=30
        )

    # 绘制趋势线
    m, b = np.polyfit(final_scores, gt_norms, 1)
    plt.plot(final_scores, m*final_scores + b, color='red', linestyle='--', linewidth=2, label=f'Trend (Corr={correlation:.2f})')

    plt.title("Motivation Proof: Base Weight Stability vs. Fine-tuning Needs")
    plt.xlabel("Your Proposed Score (High = Stable/Pre-trained Well)")
    plt.ylabel("Actual LoRA Update Norm (Fine-tuning Magnitude)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("motivation_proof_correlation.png", dpi=300)
    plt.show()
    print("图表已保存为 motivation_proof_correlation.png")

    # ================= 5. 输出排序后的建议策略 =================
    print("\n======== Top-K 独享策略建议 (基于倒数/Bottom-K) ========")
    
    # 我们根据 final_scores 从小到大排序 (Bottom-K)
    # Score 越小 -> 越不稳定 -> 需要独享 LoRA
    
    sorted_indices = np.argsort(final_scores) # 默认升序
    
    # 假设我们取前 30% 作为独享层
    top_k_count = int(len(final_scores) * 0.3)
    exclusive_indices = sorted_indices[:top_k_count]
    
    print(f"Total Modules analyzed: {len(final_scores)}")
    print(f"Selecting Bottom {top_k_count} modules for EXCLUSIVE LoRA parameters:")
    
    # 打印前10个建议独享的层信息
    print(f"{'Rank':<5} | {'Layer':<5} | {'Module':<10} | {'Score (Low=Good)':<10} | {'GT Norm (High=Need)':<10}")
    print("-" * 60)
    for i in range(20): # 打印前20个
        idx = exclusive_indices[i]
        print(f"{i+1:<5} | {raw_metrics['layer_idx'][idx]:<5} | {raw_metrics['module_type'][idx]:<10} | {final_scores[idx]:.4f}       | {gt_norms[idx]:.4f}")

if __name__ == "__main__":
    main()