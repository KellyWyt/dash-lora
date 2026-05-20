import torch
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForCausalLM
import os
import copy
import re
from scipy.stats import entropy

# ================= 配置路径 =================
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test/ablation_study_smart_prune"

# ================= ✂️ 核心配置：差异化剪枝策略 ✂️ =================
# 这里的 Ratio 指的是：在该模块类型中，剪掉 "分数最高（最稳/最冗余）" 的百分比
# 例如 0.4 表示剪掉分数最高的 40% 的层
PRUNE_CONFIG = {
    # --- 保护区 (更新幅度大或无相关性) ---
    'gate_proj': 0.0,  # 绝对不剪，它是微调主力
    'up_proj':   0.0,  # 建议保留
    'q_proj':    0.1,  # 稍微剪一点最稳的
    'k_proj':    0.1,  # 稍微剪一点最稳的
    
    # --- 信任区 (强负相关，Score 越高的确更新越小) ---
    'down_proj': 0.4,  # 剪掉 Top 40% High Score
    'o_proj':    0.4,  # 剪掉 Top 40% High Score
    
    # --- 激进区 (整体更新都很小) ---
    'v_proj':    0.5,  # 剪掉 Top 50%
}

TARGET_LAYERS = range(32)

# ================= 1. 统计计算 (去掉 Rank，只用 CPU/GPU 计算 Kurt/Ent) =================
def compute_metrics_on_gpu(weight_chunk, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    results = {}
    for name, weight in weight_chunk:
        try:
            w = weight.to(device, dtype=torch.float32)
            
            # 1. Kurtosis
            if w.std() < 1e-8:
                kurt_score = 0.0
            else:
                mean, std = w.mean(), w.std()
                kurt_val = ((w - mean) ** 4).mean() / (std ** 4) - 3
                # 归一化或是直接用原始值，后续统一归一化
                kurt_score = kurt_val.item() 
            
            # 2. Entropy
            # 为了速度，使用直方图估算
            bins = 100
            min_val, max_val = w.min(), w.max()
            if (max_val - min_val) < 1e-8:
                ent_norm = 0.0 # 极低熵 (高度确定)
            else:
                hist = torch.histc(w, bins=bins, min=min_val, max=max_val)
                hist = hist / hist.sum() # Probability
                hist = hist[hist > 0]
                ent = -torch.sum(hist * torch.log(hist))
                max_ent = torch.log(torch.tensor(bins, device=device))
                ent_norm = (ent / max_ent).item() # Normalized 0~1
            
            results[name] = {'kurt': kurt_score, 'ent': ent_norm}
            del w
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            results[name] = {'kurt': 0.0, 'ent': 1.0}
    torch.cuda.empty_cache()
    return results

# ================= 2. 辅助函数 =================
def get_keys_by_layer_and_type(adapter_keys, layer_idx, proj_type):
    matched_keys = []
    # 匹配 layers.5.mlp.gate_proj.lora_A1.weight
    pattern = re.compile(fR"layers\.{layer_idx}\..*?\.{proj_type}\.(lora_[AB]\d+)\.weight")
    for key in adapter_keys:
        if pattern.search(key):
            matched_keys.append(key)
    return matched_keys

def perform_smart_pruning(original_adapter, prune_plan, adapter_keys):
    """
    prune_plan: list of (layer_idx, proj_type)
    """
    new_adapter = copy.deepcopy(original_adapter)
    prune_stats = {"A1_Video": 0, "A2_Audio": 0, "Ignored_Text": 0, "Ignored_Shared_B": 0}
    
    for target_idx, p_type in prune_plan:
        target_keys = get_keys_by_layer_and_type(adapter_keys, target_idx, p_type)
        
        for key in target_keys:
            # === 🛡️ 保护机制 ===
            
            # 1. 保护 Text (A0)
            if "lora_A0" in key:
                prune_stats["Ignored_Text"] += 1
                continue
            
            # 2. 保护共享的 B0
            # 因为 B0 被 A0 (Text) 依赖，所以无论如何都不能剪 B0
            if "lora_B" in key:  # 匹配 lora_B0, lora_B 等
                prune_stats["Ignored_Shared_B"] += 1
                continue
                
            # === ✂️ 执行剪枝 (只针对 A1 和 A2) ===
            # 既然 B 是共享的，我们只需要把 A1 (Video) 和 A2 (Audio) 置零
            # 这样 Video/Audio 的数据流在这一层就会被阻断（退化为 Base Model），
            # 而 Text (A0 + B0) 依然正常工作。
            
            if "lora_A1" in key:
                new_adapter[key] = torch.zeros_like(original_adapter[key])
                prune_stats["A1_Video"] += 1
                
            elif "lora_A2" in key:
                new_adapter[key] = torch.zeros_like(original_adapter[key])
                prune_stats["A2_Audio"] += 1
            
    return new_adapter, prune_stats

# ================= 3. 主程序 =================
def main():
    mp.set_start_method('spawn', force=True)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Weights & Calculate Scores ---
    print(f"Loading Base Model: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    
    tasks = []
    print("Extracting weights...")
    for name, module in model.named_modules():
        for i in TARGET_LAYERS:
            if f"layers.{i}." in name:
                for t in PRUNE_CONFIG.keys(): # 只关注配置表里的模块
                    if name.endswith(t) and hasattr(module, 'weight'):
                        tasks.append((name, module.weight.data.clone()))
                break
    del model

    # GPU Parallel Calculation
    available_gpus = [i for i in range(torch.cuda.device_count())]
    num_gpus = len(available_gpus) if available_gpus else 1
    chunk_size = len(tasks) // num_gpus + 1
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    print("Calculating Kurtosis & Entropy...")
    raw_metrics = {}
    with mp.Pool(num_gpus) as pool:
        async_results = []
        for i, chunk in enumerate(chunks):
            if chunk:
                gpu_id = available_gpus[i % len(available_gpus)]
                async_results.append(pool.apply_async(compute_metrics_on_gpu, (chunk, gpu_id)))
        for res in async_results:
            raw_metrics.update(res.get())

    # --- 2. Normalize & Scoring ---
    layer_scores = [] 
    grouped_data = {t: {'names': [], 'kurt': [], 'ent': []} for t in PRUNE_CONFIG.keys()}
    
    for name, m in raw_metrics.items():
        for t in PRUNE_CONFIG.keys():
            if name.endswith(t):
                grouped_data[t]['names'].append(name)
                grouped_data[t]['kurt'].append(m['kurt'])
                grouped_data[t]['ent'].append(m['ent'])
                break
    
    for t_type, data in grouped_data.items():
        if not data['names']: continue
        k_arr = np.array(data['kurt'])
        e_arr = np.array(data['ent'])
        
        # Min-Max Normalization within Module Type
        k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-6)
        e_norm = (e_arr - e_arr.min()) / (e_arr.max() - e_arr.min() + 1e-6)
        
        # Formula: 0.6 * Kurt + 0.4 * (1 - Entropy)
        # High Score = Stable / Redundant
        scores = 0.6 * k_norm + 0.4 * (1 - e_norm)
        
        for name, score in zip(data['names'], scores):
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                lid = int(match.group(1))
                layer_scores.append({'layer': lid, 'type': t_type, 'score': score})

    # --- 3. Pruning Plan Generation ---
    print("\n>>> Generating Pruning Plan...")
    prune_plan = []
    
    for t_type, ratio in PRUNE_CONFIG.items():
        if ratio <= 0:
            print(f"  - [{t_type}] Keep All (Ratio=0)")
            continue
            
        items = [x for x in layer_scores if x['type'] == t_type]
        # Sort High to Low -> We prune High Score layers
        items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
        
        num_prune = int(len(items_sorted) * ratio)
        targets = items_sorted[:num_prune]
        
        print(f"  - [{t_type}] Pruning Top {ratio*100}% High Score Layers ({num_prune} layers)")
        for t in targets:
            prune_plan.append((t['layer'], t_type))

    # --- 4. Apply Pruning ---
    print(f"\nLoading LoRA: {ADAPTER_BIN_PATH}")
    original_adapter = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    adapter_keys = list(original_adapter.keys())
    
    new_adapter, stats = perform_smart_pruning(original_adapter, prune_plan, adapter_keys)
    
    path_out = os.path.join(OUTPUT_DIR, "exp_smart_prune.bin")
    torch.save(new_adapter, path_out)
    
    print("\n>>> Pruning Complete!")
    print(f"Saved to: {path_out}")
    print("Stats:")
    print(f"  - Pruned Video (A1/B1) Tensors: {stats['A1_Video']}")
    print(f"  - Pruned Audio (A2/B2) Tensors: {stats['A2_Audio']}")
    print(f"  - PROTECTED Text (A0) Tensors:  {stats['Ignored_Text']}")
    print(f"  - PROTECTED Shared (B0) Tensors:{stats['Ignored_Shared']}")
    
    print("\n>>> Analysis:")
    print("If Strict Acc remains high (~77%), your hypothesis is verified:")
    print("1. 'High Score' correctly identified redundant AV layers in down/o_proj.")
    print("2. Protecting gate_proj prevented collapse.")
    print("3. Protecting A0 (Text) preserved instruction following.")

if __name__ == "__main__":
    main()