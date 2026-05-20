import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
checkpoint_paths = {
    # '1:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/finetune_weights.bin",
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}
modality_key = 'lora_A1' # 重点分析 Video 专家 (A1)
shared_b_key = 'lora_B0'

# ================= 核心分析函数 =================
def calculate_update_intensity(path):
    print(f"Analyzing Update Intensity for {path}...")
    if not os.path.exists(path): return None
    state_dict = torch.load(path, map_location="cpu")
    
    norm_trained = 0.0
    norm_initial = 0.0
    count = 0
    
    for key in state_dict.keys():
        if modality_key in key and ".weight" in key:
            # 1. 获取训练后的 A 权重范数
            A_trained = state_dict[key].float()
            norm_trained += torch.norm(A_trained).item()
            
            # 2. 模拟初始化的 A 权重范数 (使用 LoRA 默认的 Kaiming Uniform)
            A_init = torch.zeros_like(A_trained)
            nn.init.kaiming_uniform_(A_init, a=np.sqrt(5))
            norm_initial += torch.norm(A_init).item()
            
            count += 1

    if count == 0: return None
    
    # 计算更新强度：训练后 / 初始
    # > 1 代表显著更新，≈ 1 代表原地踏步
    intensity = norm_trained / norm_initial
    return intensity

# ================= 执行计算 =================
intensity_results = {}
for ratio, path in checkpoint_paths.items():
    val = calculate_update_intensity(path)
    if val: intensity_results[ratio] = val

# ================= 绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(9, 6))

ratios = list(intensity_results.keys())
values = list(intensity_results.values())
x = np.arange(len(ratios))

# 1. 画更新强度柱状图
colors = ['#ff9999', '#66b3ff', '#99e699']
bars = ax.bar(x, values, color=colors, edgecolor='black', width=0.5, alpha=0.9, zorder=3)

# 2. 画“无变化”基准线 (1.0 线)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, zorder=4, label='No Update Baseline (Initial State)')

# 3. 添加数值标注
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 装饰
ax.set_ylabel("Expert Update Intensity ($||A_{trained}|| / ||A_{init}||$)")
ax.set_title("Perceptual Expert Evolution: Update Magnitude vs. Modality Ratio", fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(ratios)

# 突出显示 4:1:1 的“停滞”状态
ax.annotate('Minimal Evolution', xy=(3, values[3]), xytext=(2, values[3] + 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=12, color='darkred', fontweight='bold')

ax.legend(loc='upper right')
ax.set_ylim(0.8, max(values) * 1.15) # Y轴从0.8开始，更清楚地看到与1.0的差距

plt.tight_layout()
plt.savefig("expert_update_intensity.png", dpi=300)
print("Saved expert_update_intensity.png")