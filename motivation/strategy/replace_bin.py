import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForCausalLM
import os
import copy

# ================= 配置路径 =================
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"
# 注意：确保读取权重的路径是存在的
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
# 输出路径：如果这个文件夹不存在，脚本现在会自动创建它
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test"

TARGET_TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TARGET_LAYERS = range(32) # layer 0-31

# ================= 1. GPU 计算核心函数 =================

def compute_metrics_on_gpu(weight_chunk, gpu_id):
    """
    在指定GPU上计算原始指标 (Kurtosis 和 Entropy)
    input: List of (name, weight_tensor)
    output: {name: {'kurt': val, 'ent': val}}
    """
    device = torch.device(f'cuda:{gpu_id}')
    results = {}
    
    for name, weight in weight_chunk:
        try:
            # 移至 GPU
            w = weight.to(device, dtype=torch.float32)
            
            # --- 1. Kurtosis Calculation (Fisher) ---
            if w.std() < 1e-8:
                kurt_score = 0.0
            else:
                mean, std = w.mean(), w.std()
                # Fisher Kurtosis: E[(x-u)^4]/std^4 - 3
                kurt_val = ((w - mean) ** 4).mean() / (std ** 4) - 3
                # 变换到 0-1 方便后续处理 (Sigmoid-like transform)
                kurt_score = 1 / (1 + torch.exp(-kurt_val / 5)).item()
            
            # --- 2. Entropy Calculation (Normalized) ---
            bins = 50
            min_val, max_val = w.min(), w.max()
            if (max_val - min_val) < 1e-8:
                ent_norm = 0.5
            else:
                # 直方图
                hist = torch.histc(w, bins=bins, min=min_val, max=max_val)
                hist = hist[hist > 0]
                hist = hist / hist.sum() # 概率分布
                # Shannon Entropy
                ent = -torch.sum(hist * torch.log(hist + 1e-8))
                # Normalize by log(bins)
                max_ent = torch.log(torch.tensor(bins, device=device))
                ent_norm = (ent / max_ent).item()
            
            results[name] = {'kurt': kurt_score, 'ent': ent_norm}
            del w
            
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            results[name] = {'kurt': 0.0, 'ent': 1.0} # 默认最差值
            
    torch.cuda.empty_cache()
    return results

# ================= 2. 主逻辑 =================

def main():
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    # --- 【新增】确保输出目录存在 ---
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    # -----------------------------

    # 1. 资源检测
    available_gpus = [i for i in range(torch.cuda.device_count())]
    if not available_gpus:
        print("Warning: No GPU detected, analysis will be very slow on CPU.")
    
    print(f"Using GPUs: {available_gpus}")

    # 2. 加载基座模型 (CPU加载以节省显存)
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    # 3. 提取目标权重
    print("Extracting target weights...")
    tasks = [] # (name, weight_tensor)
    
    for name, module in model.named_modules():
        # 筛选: layers.0 ~ layers.31 且属于指定类型
        is_target_layer = False
        for i in TARGET_LAYERS:
            if f"layers.{i}." in name:
                is_target_layer = True
                break
        
        if is_target_layer:
            # 检查类型后缀
            for t_type in TARGET_TYPES:
                if name.endswith(t_type) and hasattr(module, 'weight'):
                    tasks.append((name, module.weight.data.clone())) # Clone 避免占用原模型内存
                    break
    
    # 释放模型内存
    del model
    import gc
    gc.collect()

    print(f"Total modules to analyze: {len(tasks)}")

    # 4. 并行计算原始指标
    num_gpus = len(available_gpus) if available_gpus else 1
    chunk_size = len(tasks) // num_gpus + 1
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    raw_metrics = {}
    
    if available_gpus:
        with mp.Pool(num_gpus) as pool:
            async_results = []
            for i, chunk in enumerate(chunks):
                if chunk:
                    gpu_id = available_gpus[i % len(available_gpus)]
                    async_results.append(pool.apply_async(compute_metrics_on_gpu, (chunk, gpu_id)))
            
            for res in async_results:
                raw_metrics.update(res.get())
    else:
        # CPU Fallback (简单实现，防止报错)
        print("Falling back to CPU (slow)...")
        # 这里省略 CPU 实现细节，假设你有 GPU
        pass

    # 5. 计算最终分数 (类型感知归一化)
    # 公式: Score = 0.6 * Norm(Kurt) + 0.4 * (1 - Norm(Ent))
    final_scores = {}
    
    # 按类型分组
    grouped_data = {t: {'names': [], 'kurt': [], 'ent': []} for t in TARGET_TYPES}
    
    for name, m in raw_metrics.items():
        for t in TARGET_TYPES:
            if t in name:
                grouped_data[t]['names'].append(name)
                grouped_data[t]['kurt'].append(m['kurt'])
                grouped_data[t]['ent'].append(m['ent'])
                break
    
    # 归一化并打分
    for t_type, data in grouped_data.items():
        if not data['names']: continue
        
        k_arr = np.array(data['kurt'])
        e_arr = np.array(data['ent'])
        
        # Min-Max Norm
        k_min, k_max = k_arr.min(), k_arr.max()
        e_min, e_max = e_arr.min(), e_arr.max()
        
        # 防止除零
        k_norm = (k_arr - k_min) / (k_max - k_min + 1e-6)
        e_norm = (e_arr - e_min) / (e_max - e_min + 1e-6)
        
        # 应用权重公式
        scores = 0.6 * k_norm + 0.4 * (1 - e_norm)
        
        for name, score in zip(data['names'], scores):
            final_scores[name] = score

    # ================= 6. 保存分数到 TXT =================
    score_file_path = os.path.join(OUTPUT_DIR, "sis_scores.txt")
    print(f"\nSaving all scores to {score_file_path}...")
    
    # 按分数从小到大排序 (Low Score = Unstable/Bottom)
    sorted_scores_list = sorted(final_scores.items(), key=lambda x: x[1])
    
    with open(score_file_path, "w") as f:
        f.write("SIS Strategy Scores (Sorted Ascending: Lowest Score = Most Unstable)\n")
        f.write("Formula: 0.6 * Norm(Kurtosis) + 0.4 * (1 - Norm(Entropy))\n")
        f.write("="*80 + "\n")
        for name, score in sorted_scores_list:
            f.write(f"{name}: {score:.6f}\n")
    
    print("Scores saved.")

    # ================= 7. 排名与替换逻辑 =================
    
    # (A) 每种类型的 Top 1 (Score 最大 = 最稳定)
    top_1_per_type = {}
    for t_type in TARGET_TYPES:
        type_scores = {k: v for k, v in final_scores.items() if t_type in k}
        if type_scores:
            best_module = max(type_scores.items(), key=lambda x: x[1])
            top_1_per_type[t_type] = best_module[0]
            print(f"Type {t_type:10} Top 1 (Source): {best_module[0][-30:]} (Score: {best_module[1]:.4f})")

    # (B) 全局 Bottom 20 (Score 最小 = 最不稳定)
    bottom_20_names = [x[0] for x in sorted_scores_list[:20]]
    
    print("\nBottom 20 Modules (Targets to be replaced):")
    for name in bottom_20_names:
        print(f"  {name} : {final_scores[name]:.4f}")

    # (C) 加载并修改 Adapter 权重
    print(f"\nLoading Adapter: {ADAPTER_BIN_PATH}")
    if not os.path.exists(ADAPTER_BIN_PATH):
        raise FileNotFoundError(f"Adapter file not found: {ADAPTER_BIN_PATH}")

    adapter_weights = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    new_adapter_weights = copy.deepcopy(adapter_weights)
    
    def get_lora_keys(state_dict, module_name):
        keys = []
        for k in state_dict.keys():
            if module_name in k and ('lora_A' in k or 'lora_B' in k):
                keys.append(k)
        return keys

    swap_count = 0
    for bad_module in bottom_20_names:
        current_type = next((t for t in TARGET_TYPES if t in bad_module), None)
        if not current_type: continue
        
        good_module = top_1_per_type.get(current_type)
        if not good_module: continue
        
        bad_keys = get_lora_keys(adapter_weights, bad_module)
        good_keys = get_lora_keys(adapter_weights, good_module)
        
        if not bad_keys:
            continue
            
        for bad_k in bad_keys:
            suffix = bad_k.split(bad_module)[-1]
            
            match_good_k = None
            for gk in good_keys:
                if gk.endswith(good_module + suffix):
                    match_good_k = gk
                    break
            
            if match_good_k:
                new_adapter_weights[bad_k] = adapter_weights[match_good_k].clone()
                swap_count += 1

    # 8. 保存结果
    output_bin_path = os.path.join(OUTPUT_DIR, "finetune_weights_SIS_swapped.bin")
    torch.save(new_adapter_weights, output_bin_path)
    print(f"\nSuccess! Replaced {swap_count} tensors.")
    print(f"Saved new weights to: {output_bin_path}")

if __name__ == "__main__":
    main()