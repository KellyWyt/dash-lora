import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ================= 用户配置 =================
# 请确保路径正确
checkpoint_paths = {
    # '1:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/finetune_weights.bin",
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}

# 模态 Key (请确保你的模型里是这些名字)
modality_keys = {'Text': 'lora_A0', 'Video': 'lora_A1', 'Audio': 'lora_A2'}
shared_b_key = 'lora_B0'

# ================= 核心数据收集 (回归简单暴力版) =================
def collect_layer_metrics(path, ratio_label):
    print(f"Processing {ratio_label} from {path}...")
    
    if not os.path.exists(path):
        print(f"⚠️ Error: Path does not exist: {path}")
        return []
    
    try:
        state_dict = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        return []

    # 临时存储： {layer_idx: [sim1, sim2, ...]}
    layer_data = {} 

    # 遍历所有 Key
    for key in state_dict.keys():
        # 只要包含 lora_B0，它就是我们要找的 B 矩阵
        if shared_b_key in key:
            # 提取前缀 (去除 lora_B0.weight)
            # 例如: base_model.model.layers.0.self_attn.q_proj.lora_B0.weight
            # prefix -> base_model.model.layers.0.self_attn.q_proj.
            prefix = key.split(shared_b_key)[0]
            
            # 尝试提取层 ID (用于分层统计)
            # 我们假设 key 中包含 "layers.X." 这种结构
            layer_id = -1
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_id = int(parts[i+1])
                    break
            
            if layer_id == -1: continue # 如果找不到层 ID，跳过

            # 构造对应的 A 矩阵 Key
            key_text = prefix + modality_keys['Text'] + ".weight"
            key_video = prefix + modality_keys['Audio'] + ".weight"
            # key_video = prefix + modality_keys['Video'] + ".weight"

            
            # 只有当 Text 和 Audio 的 A 矩阵都存在时，才能计算相似度
            if key_text in state_dict and key_video in state_dict:
                v1 = state_dict[key_text].float().view(-1)
                v2 = state_dict[key_video].float().view(-1)
                
                if v1.norm() > 0 and v2.norm() > 0:
                    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    
                    if layer_id not in layer_data:
                        layer_data[layer_id] = []
                    layer_data[layer_id].append(sim)

    # 计算每一层的平均相似度
    final_metrics = []
    sorted_layers = sorted(layer_data.keys())
    print(f"   Found {len(sorted_layers)} layers with valid adapter pairs.")
    
    for lid in sorted_layers:
        sims = layer_data[lid]
        if sims:
            final_metrics.append(np.mean(sims))
            
    print(f"   Collected {len(final_metrics)} layer-wise data points.")
    return final_metrics

# ================= 执行 =================
data_records = []

for ratio, path in checkpoint_paths.items():
    layer_values = collect_layer_metrics(path, ratio)
    for v in layer_values:
        data_records.append({'Ratio': ratio, 'Value': v})

if not data_records:
    print("❌ Error: No data collected! Please check your Key names (lora_A0/A1) in the bin file.")
    # 打印一些 Keys 帮助调试
    dummy_path = list(checkpoint_paths.values())[0]
    if os.path.exists(dummy_path):
        sd = torch.load(dummy_path, map_location="cpu")
        print("\n--- Debug: First 10 Keys in Checkpoint ---")
        for k in list(sd.keys())[:10]:
            print(k)
    exit()

df = pd.DataFrame(data_records)

# ================= 绘图 (Box Plot) =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.6)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(8, 6))

# 使用 try-except 确保即使 seaborn 版本不同也能画图
try:
    sns.boxplot(x='Ratio', y='Value', data=df, palette=['#ff9999', '#66b3ff', '#99e699'], width=0.5, ax=ax)
    sns.stripplot(x='Ratio', y='Value', data=df, color='black', size=4, alpha=0.3, jitter=True, ax=ax)
except:
    df.boxplot(column='Value', by='Ratio', ax=ax)

# ax.set_title("Layer-wise Directional Conflict Analysis")
ax.set_ylabel("Text-Audio Cosine Similarity")
ax.set_xlabel("Ratio Settings")

save_path = "box_plot_analysis_fixed_Audio.pdf"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"✅ Success! Saved {save_path}")

save_path1 = "box_plot_analysis_fixed_Audio.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"✅ Success! Saved {save_path1}")