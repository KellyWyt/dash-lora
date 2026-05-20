import torch
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForCausalLM
import os
import copy
import re

# ================= 配置路径 =================
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test/ablation_study_global_sharing"

# ================= 🧮 核心配置：配额变量 🧮 =================
TARGET_LAYERS = range(32)
MODULE_TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# 总模块数 = 32层 * 7种类型 = 224
TOTAL_MODULES = len(TARGET_LAYERS) * len(MODULE_TYPES)

# 分母变量 (可调整)
TEXT_DENOM = 8   # Text 独享比例 1/8
AV_DENOM   = 16  # Video/Audio 独享比例 1/16

# 计算独享名额 (Top N Norms will be Exclusive)
QUOTA_TEXT = TOTAL_MODULES // TEXT_DENOM      # 28
QUOTA_VIDEO = TOTAL_MODULES // AV_DENOM       # 14
QUOTA_AUDIO = TOTAL_MODULES // AV_DENOM       # 14

print(f"Total Modules: {TOTAL_MODULES}")
print(f"Exclusive Quotas -> Text: {QUOTA_TEXT}, Video: {QUOTA_VIDEO}, Audio: {QUOTA_AUDIO}")

# ================= 1. 计算 Base Model Norm (排序依据) =================
def compute_norm_on_gpu(weight_chunk, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    results = {}
    for name, weight in weight_chunk:
        try:
            w = weight.to(device, dtype=torch.float32)
            # 计算 L2 Norm
            norm_val = torch.norm(w, p=2).item()
            results[name] = norm_val
            del w
        except Exception as e:
            print(f"Error {name}: {e}")
            results[name] = 0.0
    torch.cuda.empty_cache()
    return results

# ================= 2. 计算全局平均矩阵 (Global Averages) =================
def compute_global_averages(adapter_state_dict):
    """
    计算每种 proj_type 下，各模态(A0, A1, A2)所有层的平均矩阵
    结构: avg_map[proj_type][lora_type] = Average_Tensor
    """
    print("Computing global averages across all layers...")
    
    # 存储求和及计数: sums[proj_type][lora_type] = tensor_sum
    sums = {}
    counts = {}
    
    pattern = re.compile(r"\.([a-z_]+_proj)\.(lora_A\d)\.weight")
    
    for key, weight in adapter_state_dict.items():
        match = pattern.search(key)
        if match:
            p_type = match.group(1) # e.g., v_proj
            l_type = match.group(2) # e.g., lora_A0
            
            if p_type not in sums:
                sums[p_type] = {}
                counts[p_type] = {}
            if l_type not in sums[p_type]:
                sums[p_type][l_type] = torch.zeros_like(weight, dtype=torch.float32)
                counts[p_type][l_type] = 0
            
            # 累加 (注意：假设同类型的矩阵形状一致，Llama2满足此条件)
            sums[p_type][l_type] += weight.float()
            counts[p_type][l_type] += 1
            
    # 计算平均值
    avg_map = {}
    for p_type in sums:
        avg_map[p_type] = {}
        for l_type in sums[p_type]:
            count = counts[p_type][l_type]
            if count > 0:
                avg_map[p_type][l_type] = sums[p_type][l_type] / count
                
    return avg_map

# ================= 3. 应用配额策略 =================
def apply_quota_sharing(original_adapter, ranked_modules, global_avgs):
    """
    ranked_modules: list of {'layer': int, 'type': str, 'norm': float}, sorted by Norm DESC
    """
    new_adapter = copy.deepcopy(original_adapter)
    stats = {
        "Text": {"Exclusive": 0, "Shared": 0},
        "Video": {"Exclusive": 0, "Shared": 0},
        "Audio": {"Exclusive": 0, "Shared": 0}
    }
    
    # 1. 确定独享名单 (Set of string identifiers "layer_idx.proj_type")
    # Norm 越大 -> 越优先独享
    
    def get_id(item): return f"{item['layer']}.{item['type']}"
    
    # 取出 Top N 作为独享
    top_text  = set([get_id(x) for x in ranked_modules[:QUOTA_TEXT]])
    top_video = set([get_id(x) for x in ranked_modules[:QUOTA_VIDEO]])
    top_audio = set([get_id(x) for x in ranked_modules[:QUOTA_AUDIO]])
    
    print(f"Top 1 Text Module (Exclusive): {ranked_modules[0]['type']} L{ranked_modules[0]['layer']} (Norm: {ranked_modules[0]['norm']:.2f})")
    print(f"Bottom 1 Module (Forced Shared): {ranked_modules[-1]['type']} L{ranked_modules[-1]['layer']} (Norm: {ranked_modules[-1]['norm']:.2f})")

    # 2. 遍历权重进行替换
    pattern = re.compile(r"layers\.(\d+)\..*?\.([a-z_]+_proj)\.(lora_A\d)\.weight")
    
    for key in new_adapter.keys():
        match = pattern.search(key)
        if match:
            lid = int(match.group(1))
            ptype = match.group(2)
            ltype = match.group(3) # lora_A0, lora_A1, lora_A2
            
            ident = f"{lid}.{ptype}"
            
            # 判断当前矩阵属于哪个模态
            # 0=Text, 1=Video, 2=Audio
            is_exclusive = False
            modality_name = ""
            
            if ltype == "lora_A0":
                modality_name = "Text"
                is_exclusive = ident in top_text
            elif ltype == "lora_A1":
                modality_name = "Video"
                is_exclusive = ident in top_video
            elif ltype == "lora_A2":
                modality_name = "Audio"
                is_exclusive = ident in top_audio
            else:
                continue # 不是 A 矩阵 (如 B)，跳过，保持原样
            
            # 执行策略
            if is_exclusive:
                # 独享：保持原样 (Do nothing)
                stats[modality_name]["Exclusive"] += 1
            else:
                # 共享：替换为全局平均值
                if ptype in global_avgs and ltype in global_avgs[ptype]:
                    # 确保类型匹配
                    avg_w = global_avgs[ptype][ltype].to(new_adapter[key].dtype)
                    new_adapter[key] = avg_w
                    stats[modality_name]["Shared"] += 1
                else:
                    print(f"Warning: No average found for {ptype} {ltype}")

    return new_adapter, stats

# ================= 4. 主程序 =================
def main():
    mp.set_start_method('spawn', force=True)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Base Model Norm Calculation ---
    print(f"Loading Base Model to calculate Norms: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    
    tasks = []
    print("Extracting weights...")
    for name, module in model.named_modules():
        for i in TARGET_LAYERS:
            if f"layers.{i}." in name:
                for t in MODULE_TYPES:
                    if name.endswith(t) and hasattr(module, 'weight'):
                        tasks.append((name, module.weight.data.clone()))
                break
    del model

    # GPU Calculation
    available_gpus = [i for i in range(torch.cuda.device_count())]
    num_gpus = len(available_gpus) if available_gpus else 1
    chunk_size = len(tasks) // num_gpus + 1
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    print("Calculating L2 Norms...")
    raw_norms = {}
    with mp.Pool(num_gpus) as pool:
        async_results = []
        for i, chunk in enumerate(chunks):
            if chunk:
                gpu_id = available_gpus[i % len(available_gpus)]
                async_results.append(pool.apply_async(compute_norm_on_gpu, (chunk, gpu_id)))
        for res in async_results:
            raw_norms.update(res.get())
            
    # --- Step 2: Global Ranking ---
    # 将 Norm 数据结构化列表
    ranked_modules = []
    for name, norm_val in raw_norms.items():
        # 解析 name: ...layers.0.self_attn.q_proj
        match = re.search(r"layers\.(\d+)\..*?\.([a-z_]+_proj)", name)
        if match:
            lid = int(match.group(1))
            ptype = match.group(2)
            ranked_modules.append({'layer': lid, 'type': ptype, 'norm': norm_val})
            
    # 按 Norm 从大到小排序 (Big Norm -> Exclusive)
    ranked_modules.sort(key=lambda x: x['norm'], reverse=True)
    
    print(f"Ranked {len(ranked_modules)} modules by Base Model L2 Norm.")

    # --- Step 3: Load Adapter & Compute Averages ---
    print(f"\nLoading LoRA: {ADAPTER_BIN_PATH}")
    original_adapter = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    
    # 计算所有层 A0, A1, A2 的平均值 (按 proj_type 分类)
    global_avgs = compute_global_averages(original_adapter)

    # --- Step 4: Apply Sharing Strategy ---
    print("\nApplying Sharing Strategy...")
    new_adapter, stats = apply_quota_sharing(original_adapter, ranked_modules, global_avgs)
    
    # Save
    path_out = os.path.join(OUTPUT_DIR, "exp_quota_sharing.bin")
    torch.save(new_adapter, path_out)
    
    print("\n>>> Process Complete!")
    print(f"Saved to: {path_out}")
    print("\n=== Statistics (Number of Layers) ===")
    print(f"Total Modules Scanned: {len(ranked_modules)}")
    
    print("\n[Text Modality A0]")
    print(f"  - Exclusive (Top {QUOTA_TEXT} Norms): {stats['Text']['Exclusive']}")
    print(f"  - Shared (Replaced by Global Mean):   {stats['Text']['Shared']}")
    
    print("\n[Video Modality A1]")
    print(f"  - Exclusive (Top {QUOTA_VIDEO} Norms): {stats['Video']['Exclusive']}")
    print(f"  - Shared (Replaced by Global Mean):   {stats['Video']['Shared']}")
    
    print("\n[Audio Modality A2]")
    print(f"  - Exclusive (Top {QUOTA_AUDIO} Norms): {stats['Audio']['Exclusive']}")
    print(f"  - Shared (Replaced by Global Mean):   {stats['Audio']['Shared']}")
    
    print("\nNote: 'Shared' here means the layer's A matrix was replaced by the average A matrix of all 32 layers for that projection type.")

if __name__ == "__main__":
    main()