import torch
import numpy as np
from transformers import AutoModelForCausalLM
import os
import copy
import re
import gc
import time

# ================= ⚙️ 路径配置 (请修改此处) =================
# 原始预训练模型路径
BASE_MODEL_PATH = "/nfs1/WYT/models/Llama-2-7b-chat-hf"

# 训练好的 LoRA 权重路径
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"

# 输出路径
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test/ablation_study_norm_based_cpu"

# ================= 🧪 核心策略配置 🧪 =================
# 逻辑：基于 "强者恒强" 定律
# - High Norm 层 (MLP) -> 更新大 -> 必须保留 (Ratio = 0.0)
# - Low Norm 层 (Attn) -> 更新小 -> 尝试剪掉 Video/Audio 特异性参数 (Ratio = 0.8)

PRUNE_CONFIG = {
    # === 🛡️ 核心知识区 (MLP) -> 坚决不剪 ===
    'gate_proj': 0.0,
    'up_proj':   0.0,
    'down_proj': 0.0,
    
    # === ✂️ 路由/关联区 (Attention) -> 激进剪枝 ===
    'q_proj':    0.8, 
    'k_proj':    0.8,
    'v_proj':    0.8,
    'o_proj':    0.8,
}

TARGET_LAYERS = range(32)

# ================= 1. CPU 计算函数 =================
def compute_norm_on_cpu(tasks):
    """
    在 CPU 上计算权重的 Frobenius Norm
    """
    results = {}
    print(f"Processing {len(tasks)} layers on CPU...")
    
    start_time = time.time()
    for idx, (name, weight) in enumerate(tasks):
        # 转换为 float32 保证精度，CPU 计算通常很快
        norm_val = torch.norm(weight.float(), p='fro').item()
        results[name] = norm_val
        
        # 简单的进度打印
        if (idx + 1) % 50 == 0:
            print(f"  - Computed {idx + 1}/{len(tasks)} layers...")
            
    print(f"Calculation finished in {time.time() - start_time:.2f} seconds.")
    return results

# ================= 2. 剪枝执行函数 (逻辑保持不变) =================
def perform_norm_based_pruning(original_adapter, prune_plan, adapter_keys):
    new_adapter = copy.deepcopy(original_adapter)
    stats = {
        "Pruned_A1_Video": 0, 
        "Pruned_A2_Audio": 0, 
        "Protected_Text_A0": 0, 
        "Protected_Shared_B": 0,
        "Protected_HighNorm_Specific": 0
    }
    
    target_set = set(prune_plan)
    
    print("Executing pruning...")
    for key in adapter_keys:
        match = re.search(r"layers\.(\d+)\..*?\.([a-z_]+_proj)", key)
        if not match:
            continue
            
        layer_idx = int(match.group(1))
        proj_type = match.group(2)
        
        is_target_layer = (layer_idx, proj_type) in target_set
        
        if "lora_A0" in key:
            stats["Protected_Text_A0"] += 1
            continue
        if "lora_B" in key:
            stats["Protected_Shared_B"] += 1
            continue
            
        if "lora_A1" in key:
            if is_target_layer:
                new_adapter[key] = torch.zeros_like(original_adapter[key])
                stats["Pruned_A1_Video"] += 1
            else:
                stats["Protected_HighNorm_Specific"] += 1
                
        elif "lora_A2" in key:
            if is_target_layer:
                new_adapter[key] = torch.zeros_like(original_adapter[key])
                stats["Pruned_A2_Audio"] += 1
            else:
                stats["Protected_HighNorm_Specific"] += 1
                
    return new_adapter, stats

# ================= 3. 主程序 =================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=======================================================")
    print("🚀 Running Norm-Based Pruning Strategy (CPU Version)")
    print("=======================================================")

    # --- Step 1: 加载 Base Model ---
    print(f"\n[1/4] Loading Base Model from: {BASE_MODEL_PATH}")
    try:
        # device_map="cpu" 显式指定使用 CPU
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            torch_dtype=torch.float32, # CPU 上使用 float32 更通用
            device_map="cpu",
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tasks = []
    print("Extracting Linear layers...")
    for name, module in model.named_modules():
        valid_layer = False
        for i in TARGET_LAYERS:
            if f"layers.{i}." in name:
                valid_layer = True
                break
        
        if valid_layer:
            for t in PRUNE_CONFIG.keys():
                if name.endswith(t) and hasattr(module, 'weight'):
                    # 必须 clone() 并 detach()，否则释放 model 时会丢失数据
                    tasks.append((name, module.weight.detach().clone()))
                    break
    
    # 立即释放大模型占用的内存
    del model
    gc.collect()
    print(f"Extracted {len(tasks)} layers. Model memory released.")

    # --- Step 2: CPU 计算 Norm ---
    print("\n[2/4] Calculating Norms (Single Process CPU)...")
    # 直接在主进程计算，避免多进程内存复制开销
    raw_norms = compute_norm_on_cpu(tasks)

    # --- Step 3: 生成剪枝计划 ---
    print("\n[3/4] Generating Pruning Plan...")
    
    layer_data = []
    for name, norm in raw_norms.items():
        match = re.search(r"layers\.(\d+)\..*?\.([a-z_]+_proj)", name)
        if match:
            lid = int(match.group(1))
            ptype = match.group(2)
            layer_data.append({'layer': lid, 'type': ptype, 'norm': norm})

    prune_plan = []
    
    for t_type, ratio in PRUNE_CONFIG.items():
        if ratio <= 0:
            print(f"  🔹 [{t_type}]: KEEP ALL (Critical High Norm Layer)")
            continue
            
        items = [x for x in layer_data if x['type'] == t_type]
        
        # 🔥 按 Norm 从小到大排序 (Low to High)
        items_sorted = sorted(items, key=lambda x: x['norm'], reverse=False)
        
        num_prune = int(len(items_sorted) * ratio)
        targets = items_sorted[:num_prune]
        
        if targets:
            max_pruned_norm = targets[-1]['norm']
            print(f"  ✂️ [{t_type}]: Pruning Bottom {ratio*100}% (Norm < {max_pruned_norm:.2f}) -> {num_prune} layers")
            for t in targets:
                prune_plan.append((t['layer'], t['type']))
        else:
            print(f"  🔹 [{t_type}]: No layers pruned (Count=0)")

    # --- Step 4: 应用剪枝 ---
    print(f"\n[4/4] Applying Pruning to LoRA Weights...")
    print(f"Loading adapter from: {ADAPTER_BIN_PATH}")
    
    original_adapter = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    adapter_keys = list(original_adapter.keys())
    
    new_adapter, stats = perform_norm_based_pruning(original_adapter, prune_plan, adapter_keys)
    
    save_filename = "exp_norm_strategy_cpu.bin"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    torch.save(new_adapter, save_path)
    
    print("\n" + "="*50)
    print(f"✅ DONE! Pruned adapter saved to:\n   {save_path}")
    print("="*50)
    print("📊 Pruning Statistics:")
    print(f"  1. [Video] A1 tensors zeroed out:   {stats['Pruned_A1_Video']}")
    print(f"  2. [Audio] A2 tensors zeroed out:   {stats['Pruned_A2_Audio']}")
    print(f"  3. [Protected] Text A0 preserved:   {stats['Protected_Text_A0']}")
    print(f"  4. [Protected] Shared B preserved:  {stats['Protected_Shared_B']}")
    print(f"  5. [Protected] High Norm Specific:  {stats['Protected_HighNorm_Specific']}")

if __name__ == "__main__":
    main()