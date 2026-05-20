import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
# 请替换为你实际的权重路径
checkpoint_paths = {
    '1:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/finetune_weights.bin",
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}
modality_keys = {'Text': 'lora_A0', 'Video': 'lora_A1', 'Audio': 'lora_A2'}
shared_b_key = 'lora_B0'
LORA_RANK = 4 # 你的 Rank 设置

# ================= 核心函数：计算真实SAR与噪声底限 =================
def calculate_sar_and_noise(path):
    print(f"Processing {path}...")
    if not os.path.exists(path): return None, None

    state_dict = torch.load(path, map_location="cpu")
    
    energy_text = 0.0
    energy_video = 0.0
    energy_noise = 0.0 # 随机噪声能量
    
    count = 0
    
    for key in state_dict.keys():
        if shared_b_key in key:
            prefix = key.split(shared_b_key)[0]
            B = state_dict[key].float()
            
            # 1. 获取 Text A (作为分母/锚点)
            key_text_A = prefix + modality_keys['Text'] + ".weight"
            if key_text_A in state_dict:
                A_text = state_dict[key_text_A].float()
                energy_text += torch.norm(B @ A_text).item()
            
            # 2. 获取 Video A (作为分子)
            key_video_A = prefix + modality_keys['Video'] + ".weight"
            if key_video_A in state_dict:
                A_video = state_dict[key_video_A].float()
                energy_video += torch.norm(B @ A_video).item()
                
                # 3. 【关键步骤】生成随机噪声基线
                # 模拟一个未训练的随机初始化矩阵 (Kaiming Uniform 或 Gaussian)
                # 保持与真实 Video A 相同的形状和设备
                A_noise = torch.randn_like(A_video) * (1.0 / np.sqrt(A_video.shape[1])) # 简单的标准差缩放
                
                # 计算这个“假”矩阵通过训练好的 B 时产生的能量
                energy_noise += torch.norm(B @ A_noise).item()
                
                count += 1
    
    if energy_text == 0: return 0, 0
    
    # 计算 SAR
    sar_video = energy_video / energy_text
    
    # 计算 噪声基线比率 (Noise-to-Anchor Ratio)
    # 这代表：如果 Video Adapter 只是随机噪声，它的 SAR 应该是多少？
    sar_noise = energy_noise / energy_text
    
    return sar_video, sar_noise

# ================= 执行计算 =================
sar_data = {}
noise_floors = [] # 存储各实验的噪声底限，取平均作为参考线

for ratio, path in checkpoint_paths.items():
    sar_val, noise_val = calculate_sar_and_noise(path)
    if sar_val is not None:
        sar_data[ratio] = sar_val
        noise_floors.append(noise_val)

# 计算平均噪声底限 (作为图上的参考线)
avg_noise_floor = np.mean(noise_floors) if noise_floors else 0.1

print(f"Calculated SAR: {sar_data}")
print(f"Calculated Noise Floor: {avg_noise_floor}")

# ================= 绘图 (带崩塌区和噪声线) =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

ratios = list(sar_data.keys())
values = list(sar_data.values())
x = np.arange(len(ratios))

# 1. 画柱状图
bars = ax.bar(x, values, color=['#ff9999', '#66b3ff', '#99e699', '#fdb462'], 
              edgecolor='black', width=0.5, alpha=0.9, zorder=3)

# 2. 【核心】画“噪声底限” (Noise Floor)
ax.axhline(y=avg_noise_floor, color='gray', linestyle='--', linewidth=2, zorder=2,
           label='Random Noise Baseline (Untrained)')

# 3. 【核心】画“崩塌区” (Collapse Zone)
# 定义崩塌区为：从 0 到 噪声线稍微往上一点 (代表虽然比噪声高一点，但也已经没用了)
# collapse_threshold = avg_noise_floor * 1.5 # 设定一个经验倍数，或者直接用 4:1:1 的值
# ax.axhspan(0, collapse_threshold, color='red', alpha=0.1, zorder=1, label='Signal Submergence Zone')

# 建议的修改方案：
ax.axhspan(0, avg_noise_floor, color='red', alpha=0.08, zorder=1, label='Signal Submergence Zone')
# ax.text(0.5, avg_noise_floor * 1.1, "COLLAPSE ZONE", color='darkred', ha='center', 
#         fontsize=14, fontweight='bold', alpha=0.4)

# 标注
ax.text(2, avg_noise_floor + 0.01, "Dead Signal Level", color='gray', ha='center', fontsize=10, fontweight='bold')
# ax.text(0.5, collapse_threshold - 0.05, "COLLAPSE ZONE", color='darkred', ha='center', fontsize=14, fontweight='bold', alpha=0.3)

# 装饰
ax.set_ylabel("Relative Signal Strength (SAR)")
# ax.set_title("Signal Submergence Analysis: Real Signal vs. Random Noise")
ax.set_xticks(x)
ax.set_xticklabels(ratios)
ax.legend(loc='upper right')
ax.set_ylim(0, max(values) * 1.2)

plt.tight_layout()
plt.savefig("sar_noise_baseline.pdf", dpi=300)
print("Saved sar_noise_baseline.png")