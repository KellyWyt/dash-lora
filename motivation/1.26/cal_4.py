import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置 =================
checkpoint_paths = {
    '1:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/finetune_weights.bin",
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}

modality_keys = {'Video': 'lora_A1'} # 我们只看 Video 怎么捣乱

def calculate_entropy(path):
    print(f"Entropy Analysis: {path}...")
    if not os.path.exists(path): return None
    state_dict = torch.load(path, map_location="cpu")
    
    all_values = []
    for key in state_dict.keys():
        if modality_keys['Video'] in key:
            # 取绝对值，归一化，算分布熵
            w = state_dict[key].float().abs().view(-1)
            all_values.append(w)
            
    if not all_values: return 0
    
    # 拼接全网所有 Video 参数
    global_w = torch.cat(all_values)
    
    # 构建直方图来模拟概率分布 P(x)
    hist = torch.histc(global_w, bins=100, min=0, max=global_w.max())
    p_x = hist / hist.sum()
    p_x = p_x[p_x > 0] # 去除0防止 log(0)
    
    # Shannon Entropy: -Sum(p * log(p))
    entropy = -torch.sum(p_x * torch.log(p_x)).item()
    return entropy

# ================= 执行 =================
entropies = {}
for ratio, path in checkpoint_paths.items():
    val = calculate_entropy(path)
    if val: entropies[ratio] = val

# ================= 绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
fig, ax = plt.subplots(figsize=(8, 6))

x = list(entropies.keys())
y = list(entropies.values())

bars = ax.bar(x, y, color=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.9, edgecolor='black')

# 标注
ax.set_ylabel("Parameter Distribution Entropy (Disorder)")
ax.set_title("Why 2:1:1 Fails? (High Entropy = Semantic Noise)")

# 在 2:1:1 上方标注 "High Disorder"
ax.text(0, y[0]*1.01, "High Disorder\n(Semantic Dilution)", ha='center', color='red', fontweight='bold')
# 在 3:1:1 上方标注 "Focused"
ax.text(1, y[1]*1.01, "Optimal Focus", ha='center', color='green', fontweight='bold')

plt.tight_layout()
plt.savefig("entropy_analysis.png", dpi=300)
print("Done. Saved entropy_analysis.png")