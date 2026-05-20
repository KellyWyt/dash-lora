import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForCausalLM
import os
import copy
import re

# ================= 配置路径 (请修改这里) =================
# 1. 基座模型路径 (用于计算 SIS 分数，验证“基座稳定性 -> LoRA冗余”的假设)
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"

# 2. 训练好的权重文件路径
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"

# 3. 输出目录
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test/ablation_study"

# 4. 替换比例 (例如 0.25 表示替换前 25% 或后 25%)
REPLACE_RATIO = 0.25 

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

# ================= 2. 智能替换辅助函数 =================

def get_keys_by_layer_and_type(adapter_keys, layer_idx, proj_type):
    """
    从所有 keys 中找到属于特定层和特定 proj_type 的所有 LoRA key。
    例如：找到 layer.0 的 q_proj 对应的 lora_A0, lora_A1, lora_A2, lora_B0
    """
    matched_keys = []
    # 正则表达式匹配：确保精确匹配 layers.{layer_idx}. ... .{proj_type}.
    # 比如: base_model.model.model.layers.0.self_attn.q_proj.lora_A0.weight
    pattern = re.compile(fR"layers\.{layer_idx}\..*?\.{proj_type}\.(lora_[AB]\d+)\.weight")
    
    for key in adapter_keys:
        match = pattern.search(key)
        if match:
            # 返回 (完整key, 后缀部分)
            # 后缀部分如: lora_A0.weight
            suffix = match.group(1) + ".weight"
            matched_keys.append((key, suffix))
    return matched_keys

def perform_replacement(original_adapter, replace_plan, adapter_keys):
    """
    执行替换操作
    replace_plan: list of (target_layer_idx, source_layer_idx, proj_type)
    """
    new_adapter = copy.deepcopy(original_adapter)
    replace_count = 0
    
    for target_idx, source_idx, p_type in replace_plan:
        if target_idx == source_idx:
            continue

        # 1. 找到源层的所有相关 keys (Expert)
        source_pairs = get_keys_by_layer_and_type(adapter_keys, source_idx, p_type)
        # source_pairs: list of (full_key, suffix) -> [('...layers.10...A0...', 'lora_A0.weight'), ...]

        # 2. 找到目标层的所有相关 keys (To be replaced)
        target_pairs = get_keys_by_layer_and_type(adapter_keys, target_idx, p_type)

        # 3. 根据后缀进行匹配替换
        # 我们要把 target 的 A0 换成 source 的 A0，target 的 B0 换成 source 的 B0
        for t_key, t_suffix in target_pairs:
            # 在 source 中寻找具有相同后缀的 key
            s_key = next((s_k for s_k, s_suf in source_pairs if s_suf == t_suffix), None)
            
            if s_key:
                # 核心替换逻辑：复制 Tensor
                new_adapter[t_key] = original_adapter[s_key].clone()
                replace_count += 1
            else:
                print(f"[Warning] Source layer {source_idx} missing suffix {t_suffix} for target layer {target_idx}")

    return new_adapter, replace_count

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
    # 建立 layer_idx 到 full_name 的映射，方便后续查表
    layer_map = {} # {'q_proj_0': score, ...}
    
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
    # 提取 (layer_idx, proj_type) -> score
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
            # 从 name (model.layers.5.self_attn.q_proj) 解析 layer_idx
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

    # --- Step 3: 确定 Source (每种类型中 Score 最高的 Layer) ---
    best_source_map = {} # {'q_proj': 12, 'k_proj': 5, ...} (Key is proj_type, Value is Layer Idx)
    
    for t_type in TARGET_TYPES:
        # 找到该类型所有层的分数
        items = [x for x in layer_scores if x['type'] == t_type]
        if items:
            # 按分数降序排列
            items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
            best_layer_idx = items_sorted[0]['layer']
            best_source_map[t_type] = best_layer_idx
            print(f"[{t_type}] Best Layer (Source): {best_layer_idx} (Score: {items_sorted[0]['score']:.4f})")

    # --- Step 4: 生成 Experiment A (High Score -> Redundant) ---
    print("\n>>> Generating Experiment A: Replacing HIGH score layers...")
    plan_A = [] # (target, source, type)
    
    for t_type in TARGET_TYPES:
        items = [x for x in layer_scores if x['type'] == t_type]
        # 降序排列 (High to Low)
        items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
        
        # 选取前 25% (High Score)
        num_replace = int(len(items_sorted) * REPLACE_RATIO)
        targets = items_sorted[:num_replace] # 这些是分数高的层
        
        source_idx = best_source_map[t_type]
        
        for target in targets:
            target_idx = target['layer']
            # 不要自己替换自己
            if target_idx != source_idx:
                plan_A.append((target_idx, source_idx, t_type))

    adapter_A, count_A = perform_replacement(original_adapter, plan_A, adapter_keys)
    path_A = os.path.join(OUTPUT_DIR, "exp_A_replace_high_score.bin")
    torch.save(adapter_A, path_A)
    print(f"Saved Exp A to {path_A}")
    print(f"  - Replaced High Score Layers with Layer {source_idx} (Best)")
    print(f"  - Total tensors replaced: {count_A} (Includes A0, A1, A2, B0 etc.)")

    # --- Step 5: 生成 Experiment B (Low Score -> Critical) ---
    print("\n>>> Generating Experiment B: Replacing LOW score layers...")
    plan_B = []
    
    for t_type in TARGET_TYPES:
        items = [x for x in layer_scores if x['type'] == t_type]
        # 升序排列 (Low to High)，这样前 25% 就是 Low Score
        items_sorted = sorted(items, key=lambda x: x['score'], reverse=False)
        
        # 选取前 25% (Low Score)
        num_replace = int(len(items_sorted) * REPLACE_RATIO)
        targets = items_sorted[:num_replace]
        
        source_idx = best_source_map[t_type] # 依然用最好的层来替换它们
        
        for target in targets:
            target_idx = target['layer']
            if target_idx != source_idx:
                plan_B.append((target_idx, source_idx, t_type))

    adapter_B, count_B = perform_replacement(original_adapter, plan_B, adapter_keys)
    path_B = os.path.join(OUTPUT_DIR, "exp_B_replace_low_score.bin")
    torch.save(adapter_B, path_B)
    print(f"Saved Exp B to {path_B}")
    print(f"  - Replaced Low Score Layers with Layer {source_idx} (Best)")
    print(f"  - Total tensors replaced: {count_B}")

if __name__ == "__main__":
    main()