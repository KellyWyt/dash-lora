import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载权重
bin_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"
print(f"正在加载 {bin_path} ...")
state_dict = torch.load(bin_path, map_location="cpu")

# 定义参数
num_layers = 32
modalities = ['A0 (Text/Default)', 'A1 (Video?)', 'A2 (Audio?)'] # 这里的名字你可以根据实际情况修改
modality_keys = ['lora_A0', 'lora_A1', 'lora_A2']
shared_b_key = 'lora_B0'

# 存储结果：shape = [3, 32] -> [模态数, 层数]
layer_importance = {m: np.zeros(num_layers) for m in modalities}

print("开始逐层分析模态重要性...")

for i in range(num_layers):
    print(f"Processing Layer {i}/{num_layers}...", end='\r')
    
    # 定义该层的前缀，例如 "base_model.model.model.layers.0."
    # 注意：PeftModel 保存的 key 通常包含 "base_model.model.model.layers"
    # 我们用更通用的匹配方式
    layer_pattern = f"layers.{i}."
    
    # 找到该层所有的模块 (q_proj, k_proj, etc.)
    # 我们先获取所有包含该层前缀的 Key
    layer_keys = [k for k in state_dict.keys() if layer_pattern in k]
    
    # 这一层里有哪些 Linear 模块被 LoRA 了？
    # 通过寻找包含 lora_B0 的 key 来确定模块前缀
    # 例如: ...layers.0.self_attn.q_proj.lora_B0.weight
    module_prefixes = set()
    for k in layer_keys:
        if shared_b_key in k:
            # 截取到 lora_B0 之前的部分作为模块前缀
            # e.g., "base_model.model.model.layers.0.self_attn.q_proj."
            prefix = k.split(shared_b_key)[0]
            module_prefixes.add(prefix)
            
    # 对这一层的每一个模块进行计算，并累加到该层的总分里
    for prefix in module_prefixes:
        # 获取共享的 B
        key_B = prefix + f"{shared_b_key}.weight"
        if key_B not in state_dict: continue
        
        B = state_dict[key_B].float()
        
        # 针对三个模态分别计算
        for m_idx, m_name in enumerate(modalities):
            key_A_suffix = modality_keys[m_idx] # e.g., lora_A0
            key_A = prefix + f"{key_A_suffix}.weight"
            
            if key_A in state_dict:
                A = state_dict[key_A].float()
                
                # 计算 Delta W = B @ A
                # 注意维度: B (out, r), A (r, in) -> B @ A (out, in)
                delta_W = torch.matmul(B, A)
                
                # 计算 Frobenius Norm 并累加
                norm = torch.norm(delta_W).item()
                layer_importance[m_name][i] += norm

print("\n分析完成！生成图表中...")

# --- 可视化设计 ---
plt.figure(figsize=(14, 7))

# 绘制折线图
markers = ['o', 's', '^']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝、橙、绿

for idx, m_name in enumerate(modalities):
    plt.plot(
        range(num_layers), 
        layer_importance[m_name], 
        marker=markers[idx], 
        color=colors[idx], 
        linewidth=2, 
        label=m_name,
        alpha=0.8
    )

plt.title("Modality-Specific Layer Importance (Based on Delta-Weight Norm)", fontsize=16)
plt.xlabel("Llama Layer Index (0-31)", fontsize=14)
plt.ylabel("Importance Score (Aggregated Norm)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# 添加背景注释（帮助论文解释）
plt.axvspan(0, 5, color='gray', alpha=0.1, label='Shallow Layers')
plt.axvspan(26, 31, color='gray', alpha=0.1, label='Deep Layers')

plt.tight_layout()
plt.savefig("motivation_multimodal_layers.png", dpi=300)
plt.show()