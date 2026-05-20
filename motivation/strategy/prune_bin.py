import torch
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForCausalLM
import os
import copy
import re

# ================= 配置路径 (请修改这里) =================
# 1. 基座模型路径 (用于计算 SIS 分数)
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"

# 2. 训练好的权重文件路径
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"

# 3. 输出目录
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test/ablation_study_pruning"

# 4. 剪枝比例 (例如 0.25 表示剪掉前 25% 或后 25%)
PRUNE_RATIO = 0.25 

TARGET_TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TARGET_LAYERS = range(32)

# ================= 1. 计算核心 (保持不变) =================
def compute_metrics_on_gpu(weight_chunk, gpu_id):
    """在 GPU 上计算 Kurtosis 和 Entropy"""
    device = torch.device(f'cuda:{gpu_id}')
    results = {}
    for name, weight in weight_chunk:
        try:
            w = weight.to(device, dtype=torch.float32)
            # Kurtosis
            if w.std() < 1e-8:
                kurt_score = 0.0
            else:
                mean, std = w.mean(), w.std()
                kurt_val = ((w - mean) ** 4).mean() / (std ** 4) - 3
                kurt_score = 1 / (1 + torch.exp(-kurt_val / 5)).item()
            
            # Entropy
            bins = 50
            min_val, max_val = w.min(), w.max()
            if (max_val - min_val) < 1e-8:
                ent_norm = 0.5
            else:
                hist = torch.histc(w, bins=bins, min=min_val, max=max_val)
                hist = hist[hist > 0]
                hist = hist / hist.sum()
                ent = -torch.sum(hist * torch.log(hist + 1e-8))
                max_ent = torch.log(torch.tensor(bins, device=device))
                ent_norm = (ent / max_ent).item()
            
            results[name] = {'kurt': kurt_score, 'ent': ent_norm}
            del w
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            results[name] = {'kurt': 0.0, 'ent': 1.0}
    torch.cuda.empty_cache()
    return results

# ================= 2. 智能剪枝辅助函数 =================

def get_keys_by_layer_and_type(adapter_keys, layer_idx, proj_type):
    """
    从所有 keys 中找到属于特定层和特定 proj_type 的所有 LoRA key。
    例如：找到 layer.0 的 q_proj 对应的 lora_A0, lora_A1, lora_A2, lora_B0
    """
    matched_keys = []
    # 正则表达式匹配：确保精确匹配 layers.{layer_idx}. ... .{proj_type}.
    pattern = re.compile(fR"layers\.{layer_idx}\..*?\.{proj_type}\.(lora_[AB]\d+)\.weight")
    
    for key in adapter_keys:
        if pattern.search(key):
            matched_keys.append(key)
    return matched_keys

def perform_pruning(original_adapter, prune_plan, adapter_keys):
    """
    执行剪枝操作：将指定层的 Tensor 全部置为 0
    prune_plan: list of (target_layer_idx, proj_type)
    """
    new_adapter = copy.deepcopy(original_adapter)
    prune_count = 0
    
    for target_idx, p_type in prune_plan:
        # 1. 找到目标层的所有相关 keys
        target_keys = get_keys_by_layer_and_type(adapter_keys, target_idx, p_type)

        if not target_keys:
            # print(f"[Warning] No keys found for Layer {target_idx} {p_type}")
            continue

        # 2. 将这些 Key 对应的 Tensor 置为全 0
        for key in target_keys:
            # 保持 shape 和 dtype 不变，只是数值变 0
            new_adapter[key] = torch.zeros_like(original_adapter[key])
            prune_count += 1
            
    return new_adapter, prune_count

# ================= 3. 主程序 =================

def main():
    mp.set_start_method('spawn', force=True)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: 计算基座模型的分数 (Base Model Analysis) ---
    print(f"Loading Base Model: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    
    print("Extracting Base Model weights for scoring...")
    tasks = []
    
    for name, module in model.named_modules():
        # 筛选 layers.0 ~ layers.31
        for i in TARGET_LAYERS:
            if f"layers.{i}." in name:
                for t_type in TARGET_TYPES:
                    if name.endswith(t_type) and hasattr(module, 'weight'):
                        tasks.append((name, module.weight.data.clone()))
                break
    del model
    
    # 并行计算指标
    available_gpus = [i for i in range(torch.cuda.device_count())]
    num_gpus = len(available_gpus) if available_gpus else 1
    chunk_size = len(tasks) // num_gpus + 1
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    print("Calculating SIS metrics on GPUs...")
    raw_metrics = {}
    with mp.Pool(num_gpus) as pool:
        async_results = []
        for i, chunk in enumerate(chunks):
            if chunk:
                gpu_id = available_gpus[i % len(available_gpus)]
                async_results.append(pool.apply_async(compute_metrics_on_gpu, (chunk, gpu_id)))
        for res in async_results:
            raw_metrics.update(res.get())

    # 归一化并计算最终分数
    layer_scores = [] # list of {'layer': 0, 'type': 'q_proj', 'score': 0.9}
    
    grouped_data = {t: {'names': [], 'kurt': [], 'ent': []} for t in TARGET_TYPES}
    for name, m in raw_metrics.items():
        for t in TARGET_TYPES:
            if name.endswith(t):
                grouped_data[t]['names'].append(name)
                grouped_data[t]['kurt'].append(m['kurt'])
                grouped_data[t]['ent'].append(m['ent'])
                break
    
    # 计算 Score
    for t_type, data in grouped_data.items():
        if not data['names']: continue
        k_arr = np.array(data['kurt'])
        e_arr = np.array(data['ent'])
        # Norm
        k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-6)
        e_norm = (e_arr - e_arr.min()) / (e_arr.max() - e_arr.min() + 1e-6)
        scores = 0.6 * k_norm + 0.4 * (1 - e_norm)
        
        for name, score in zip(data['names'], scores):
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                lid = int(match.group(1))
                layer_scores.append({'layer': lid, 'type': t_type, 'score': score})

    print("SIS Scores calculated on Base Model.")

    # --- Step 2: 准备 LoRA 权重 ---
    print(f"Loading Finetuned Weights: {ADAPTER_BIN_PATH}")
    if not os.path.exists(ADAPTER_BIN_PATH):
        raise FileNotFoundError(f"File not found: {ADAPTER_BIN_PATH}")
    
    original_adapter = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    adapter_keys = list(original_adapter.keys())

    # --- Step 4: 生成 Experiment A (Prune HIGH Score -> Validate Redundancy) ---
    print(f"\n>>> Generating Experiment A: Pruning TOP {PRUNE_RATIO*100}% HIGH score layers...")
    plan_A = [] # (target, type)
    
    for t_type in TARGET_TYPES:
        items = [x for x in layer_scores if x['type'] == t_type]
        # 降序排列 (High to Low) -> High Score layers first
        items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
        
        # 选取前 25% (High Score) 进行剪枝
        num_prune = int(len(items_sorted) * PRUNE_RATIO)
        targets = items_sorted[:num_prune]
        
        for target in targets:
            plan_A.append((target['layer'], t_type))

    adapter_A, count_A = perform_pruning(original_adapter, plan_A, adapter_keys)
    path_A = os.path.join(OUTPUT_DIR, "exp_A_prune_high_score.bin")
    torch.save(adapter_A, path_A)
    print(f"Saved Exp A to {path_A}")
    print(f"  - Action: Set tensors to ZERO for layers with HIGH importance scores (Expected to perform WELL if High Score = Redundant)")
    print(f"  - Total tensors pruned (zeroed): {count_A}")

    # --- Step 5: 生成 Experiment B (Prune LOW Score -> Validate Importance) ---
    print(f"\n>>> Generating Experiment B: Pruning TOP {PRUNE_RATIO*100}% LOW score layers...")
    plan_B = []
    
    for t_type in TARGET_TYPES:
        items = [x for x in layer_scores if x['type'] == t_type]
        # 升序排列 (Low to High) -> Low Score layers first
        items_sorted = sorted(items, key=lambda x: x['score'], reverse=False)
        
        # 选取前 25% (Low Score) 进行剪枝
        num_prune = int(len(items_sorted) * PRUNE_RATIO)
        targets = items_sorted[:num_prune]
        
        for target in targets:
            plan_B.append((target['layer'], t_type))

    adapter_B, count_B = perform_pruning(original_adapter, plan_B, adapter_keys)
    path_B = os.path.join(OUTPUT_DIR, "exp_B_prune_low_score.bin")
    torch.save(adapter_B, path_B)
    print(f"Saved Exp B to {path_B}")
    print(f"  - Action: Set tensors to ZERO for layers with LOW importance scores (Expected to CRASH if Low Score = Critical)")
    print(f"  - Total tensors pruned (zeroed): {count_B}")

if __name__ == "__main__":
    main()