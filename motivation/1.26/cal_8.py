import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
checkpoint_paths = {
    '1:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/finetune_weights.bin",
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}
modality_keys = {'Text': 'lora_A0', 'Video': 'lora_A1', 'Audio': 'lora_A2'}
shared_b_key = 'lora_B0'
LORA_RANK = 4 

# ================= 核心计算函数：三级能量分析 =================
def calculate_complex_sar_metrics(path):
    print(f"Processing {path}...")
    if not os.path.exists(path): return None
    state_dict = torch.load(path, map_location="cpu")
    
    e_real_text, e_real_video = 0.0, 0.0
    e_randA_trainedB = 0.0     # 随机A + 训练B (局部噪声)
    e_randA_randB_v = 0.0      # 随机A + 随机B (全局初始状态-视频端)
    e_randA_randB_t = 0.0      # 随机A + 随机B (全局初始状态-文本端)
    
    for key in state_dict.keys():
        if shared_b_key in key:
            prefix = key.split(shared_b_key)[0]
            B_trained = state_dict[key].float()
            
            # 获取对应的 A 矩阵 Key
            key_t_A = prefix + modality_keys['Text'] + ".weight"
            key_v_A = prefix + modality_keys['Video'] + ".weight"
            
            if key_t_A in state_dict and key_v_A in state_dict:
                A_t = state_dict[key_t_A].float()
                A_v = state_dict[key_v_A].float()
                
                # 1. 真实信号能量
                e_real_text += torch.norm(B_trained @ A_t).item()
                e_real_video += torch.norm(B_trained @ A_v).item()
                
                # 2. 局部噪声基准 (随机A + 训练B)
                # 使用 Kaiming 分布的标准差进行缩放
                A_noise_v = torch.randn_like(A_v) * (1.0 / np.sqrt(A_v.shape[1]))
                e_randA_trainedB += torch.norm(B_trained @ A_noise_v).item()
                
                # 3. 全局随机基准 (随机A + 随机B)
                # 模拟完全未训练的状态
                B_rand = torch.randn_like(B_trained) * (1.0 / np.sqrt(B_trained.shape[1]))
                A_rand_v = torch.randn_like(A_v) * (1.0 / np.sqrt(A_v.shape[1]))
                A_rand_t = torch.randn_like(A_t) * (1.0 / np.sqrt(A_t.shape[1]))
                
                e_randA_randB_v += torch.norm(B_rand @ A_rand_v).item()
                e_randA_randB_t += torch.norm(B_rand @ A_rand_t).item()

    # 计算各项比率 (SAR)
    sar_real = e_real_video / e_real_text
    sar_local_noise = e_randA_trainedB / e_real_text
    sar_global_rand = e_randA_randB_v / e_randA_randB_t # 理论上接近 1.0
    
    return sar_real, sar_local_noise, sar_global_rand

# ================= 数据采集 =================
results = {}
for ratio, path in checkpoint_paths.items():
    res = calculate_complex_sar_metrics(path)
    if res: results[ratio] = res

# 格式化数据
ratios = list(results.keys())
real_sars = [results[r][0] for r in ratios]
local_noises = [results[r][1] for r in ratios]
# 全局随机状态在各实验中非常稳定，取平均作为一条全局线
global_rand_baseline = np.mean([results[r][2] for r in ratios]) 

print(f"\n[Results] Real SARs: {real_sars}")
print(f"[Results] Local Noise (Rand A + Trained B): {local_noises}")
print(f"[Results] Global Random (Rand A + Rand B): {global_rand_baseline}")

# ================= 绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(ratios))

# 1. 画主柱状图 (真实信号强度)
bars = ax.bar(x, real_sars, color=['#ff9999', '#66b3ff', '#99e699', '#fdb462'], 
              edgecolor='black', width=0.5, alpha=0.9, zorder=3, label='Real Adapter Signal')

# 2. 画每一个比例下的“局部噪声线” (随机A + 训练B)
# 以短横线的形式悬浮在柱子上方/内部，视觉对比极强
for i in range(len(ratios)):
    ax.hlines(y=local_noises[i], xmin=i-0.25, xmax=i+0.25, color='black', 
              linestyle='--', linewidth=2, zorder=4, 
              label='Local Noise (Rand A, Trained B)' if i == 0 else "")

# 3. 画“全局随机线” (随机A + 随机B)
# 代表未训练模型的初始状态
ax.axhline(y=global_rand_baseline, color='red', linestyle='-.', linewidth=2, zorder=2,
           label='Global Random State (Initial)')

# 4. 信号淹没区阴影
avg_local_noise = np.mean(local_noises)
ax.axhspan(0, avg_local_noise, color='red', alpha=0.08, zorder=1, label='Signal Submergence Zone')

# 标注
ax.text(3.3, global_rand_baseline + 0.02, "Untrained Ceiling", color='red', ha='right', fontsize=10, fontweight='bold')
ax.text(3.3, avg_local_noise - 0.05, "Dead Signal Level", color='gray', ha='right', fontsize=10, fontweight='bold')

# 装饰
ax.set_ylabel("Relative Signal Strength (SAR)")
ax.set_xticks(x)
ax.set_xticklabels(ratios)
ax.legend(loc='upper right', frameon=True, fontsize='small')
ax.set_ylim(0, max(real_sars + [global_rand_baseline]) * 1.2)

plt.tight_layout()
plt.savefig("multi_level_sar_diagnostic.png", dpi=300)
print("Saved multi_level_sar_diagnostic.png")