import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
# 请替换为你实际的三个权重文件路径
checkpoint_paths = {
    '1:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/finetune_weights.bin",
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}

# 模态 Key (根据你的实际情况修改)
modality_keys = {
    'Text': 'lora_A0',
    'Video': 'lora_A1', 
    'Audio': 'lora_A2'
}
shared_b_key = 'lora_B0'

# ================= 核心计算函数 =================
def calculate_dominance(path):
    print(f"Processing {path}...")
    try:
        state_dict = torch.load(path, map_location="cpu")
    except:
        print(f"Error loading {path}")
        return None

    # 存储每个模态的总能量 (L2 Norm)
    energy_sums = {'Text': 0.0, 'Video': 0.0, 'Audio': 0.0}
    
    for key in state_dict.keys():
        if shared_b_key in key: 
            prefix = key.split(shared_b_key)[0]
            B = state_dict[key].float()
            
            for m_name, m_key in modality_keys.items():
                full_a_key = prefix + m_key + ".weight"
                if full_a_key in state_dict:
                    A = state_dict[full_a_key].float()
                    # 计算 ΔW = B @ A
                    # 注意：在 r=4 时，Magnitude 是唯一的信息强度指标
                    delta_W = torch.matmul(B, A)
                    norm = torch.norm(delta_W).item()
                    energy_sums[m_name] += norm
                    
    return energy_sums

# ================= 执行分析 =================
ratios_data = {}
absolute_energies = {'Text': [], 'Video': [], 'Audio': []} # 用于调试

for ratio_name, path in checkpoint_paths.items():
    energies = calculate_dominance(path)
    if energies:
        # 计算相对强度 (Signal-to-Anchor Ratio, SAR)
        # 意思是在 Text 这个巨大的 Anchor 面前，感知信号有多强？
        sar_video = energies['Video'] / (energies['Text'] + 1e-6)
        sar_audio = energies['Audio'] / (energies['Text'] + 1e-6)
        
        ratios_data[ratio_name] = {'Video': sar_video, 'Audio': sar_audio}
        
        # 记录绝对值用于观察
        absolute_energies['Text'].append(energies['Text'])

# ================= 绘图 (学术风) =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.6)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

labels = list(ratios_data.keys()) # ['2:1:1', '3:1:1', '4:1:1']
video_vals = [ratios_data[k]['Video'] for k in labels]
audio_vals = [ratios_data[k]['Audio'] for k in labels]

x = np.arange(len(labels))
width = 0.3

# 绘制感知信号强度
rects1 = ax.bar(x - width/2, video_vals, width, label='Video Signal Strength (vs. Text)', color='#87CEEB', edgecolor='black', alpha=0.9, hatch='//')
rects2 = ax.bar(x + width/2, audio_vals, width, label='Audio Signal Strength (vs. Text)', color='#98FB98', edgecolor='black', alpha=0.9, hatch='\\\\')

# 动态计算并绘制“崩塌阈值” (Collapse Threshold)
# 逻辑：3:1:1 是好的，4:1:1 是崩的。所以阈值一定在它们中间。
# 我们取 3:1:1 和 4:1:1 的中间值作为 "Critical Line"
threshold_y = (video_vals[1] + video_vals[2]) / 2 
plt.axhline(y=threshold_y, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
plt.text(2.6, threshold_y + 0.01, "Critical Survival Threshold", color='red', fontsize=12, fontweight='bold', ha='right')

# 标注崩塌区域
plt.axvspan(1.5, 2.5, color='gray', alpha=0.1)
plt.text(2, max(video_vals)*0.9, "Signal Submergence\n(Accuracy Collapse)", ha='center', fontsize=12, color='darkred', fontweight='bold')

# 设置轴
ax.set_ylabel('Relative Signal Strength (SAR)')
ax.set_title('Why 4:1:1 Fails? (Signal Submergence at r=4)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(video_vals) * 1.2)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("signal_submergence_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# 打印数值供论文引用
print(f"2:1:1 Video SAR: {video_vals[0]:.4f}")
print(f"3:1:1 Video SAR: {video_vals[1]:.4f}")
print(f"4:1:1 Video SAR: {video_vals[2]:.4f}")
print(f"Drop from 3:1 to 4:1: {(video_vals[1] - video_vals[2])/video_vals[1]*100:.2f}%")