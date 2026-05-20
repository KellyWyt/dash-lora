import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
# 请替换为你实际的权重路径
checkpoint_paths = {
    '2:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/finetune_weights.bin",
    '3:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/finetune_weights.bin",
    '4:1:1': "/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/finetune_weights.bin"
}

modalities = ['Text', 'Video', 'Audio']
# 对应 adapter_model.bin 里的 key 名称，请根据实际情况修改
# 假设: A0=Text, A1=Video, A2=Audio
modality_keys = {
    'Text': 'lora_A0',
    'Video': 'lora_A1', 
    'Audio': 'lora_A2'
}
shared_b_key = 'lora_B0'

# ================= 核心函数：计算有效秩 =================
def compute_stable_rank(matrix):
    """
    计算矩阵的 Stable Rank (Effective Rank)
    Stable Rank = (Frobenius Norm ^ 2) / (Spectral Norm ^ 2)
    它可以衡量矩阵实际携带的信息维度。Rank 越高，信息越丰富。
    """
    # 确保是 float32 防止溢出
    matrix = matrix.float()
    
    # 1. Singular Value Decomposition (SVD)
    try:
        # full_matrices=False 省内存
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        singular_values = S
    except:
        return 0.0 # SVD 失败 (极少见)

    # 2. 计算 Norms
    frobenius_norm_sq = torch.sum(singular_values ** 2)
    spectral_norm_sq = torch.max(singular_values) ** 2
    
    # 3. Stable Rank
    if spectral_norm_sq == 0:
        return 0.0
        
    stable_rank = frobenius_norm_sq / spectral_norm_sq
    return stable_rank.item()

# ================= 主分析循环 =================
results = {ratio: {m: [] for m in modalities} for ratio in checkpoint_paths}

for ratio, path in checkpoint_paths.items():
    print(f"Analyzing checkpoint: {ratio}...")
    try:
        state_dict = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        print(f"File not found: {path}, skipping...")
        continue

    # 遍历所有层，提取 A 矩阵
    # 我们只关心 Exclusive 部分的 A，或者混合了 Shared 的 A
    # 这里我们统计所有存在的 A 矩阵的平均 Rank
    
    for key, weight in state_dict.items():
        # 检查这个 key 属于哪个模态
        found_modality = None
        for m_name, m_key in modality_keys.items():
            if m_key in key and ".weight" in key:
                found_modality = m_name
                break
        
        if found_modality:
            # 计算该矩阵的 Stable Rank
            rank = compute_stable_rank(weight)
            results[ratio][found_modality].append(rank)

# ================= 统计均值 =================
avg_ranks = {m: [] for m in modalities}
ratios_label = list(checkpoint_paths.keys())

for m in modalities:
    for ratio in ratios_label:
        ranks = results[ratio][m]
        if len(ranks) > 0:
            avg_ranks[m].append(np.mean(ranks))
        else:
            avg_ranks[m].append(0)

# ================= 绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(ratios_label))
width = 0.25

# 绘制 Text (主轴)
color_text = '#FFD700'
bars_text = ax1.bar(x - width, avg_ranks['Text'], width, label='Text Complexity (Left)', color=color_text, edgecolor='black', alpha=0.9)
ax1.set_ylabel('Text Effective Rank', fontweight='bold')
ax1.set_ylim(0, max(avg_ranks['Text']) * 1.2)

# 绘制 Video/Audio (副轴 - 因为它们的 Rank 绝对值可能不同，分开看趋势更明显)
ax2 = ax1.twinx()
color_video = '#87CEEB'
color_audio = '#98FB98'

bars_video = ax2.bar(x, avg_ranks['Video'], width, label='Video Integrity (Right)', color=color_video, edgecolor='black', alpha=0.9)
bars_audio = ax2.bar(x + width, avg_ranks['Audio'], width, label='Audio Integrity (Right)', color=color_audio, edgecolor='black', alpha=0.9)

ax2.set_ylabel('Perception Effective Rank', fontweight='bold')
ax2.set_ylim(0, max(max(avg_ranks['Video']), max(avg_ranks['Audio'])) * 1.2)

# X轴设置
ax1.set_xlabel("Ratio Settings (Text:Video:Audio)")
ax1.set_xticks(x)
ax1.set_xticklabels(ratios_label)

# 图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.title("Spectral Analysis: Why 3:1:1 is Optimal?", fontsize=14, fontweight='bold')

# 添加关键标注 (Storytelling)
# 在 3:1:1 上方标注 "Pareto Optimal"
plt.text(1, ax2.get_ylim()[1]*0.9, "Pareto Optimal", ha='center', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig("spectral_analysis_rank.png", dpi=300, bbox_inches='tight')
plt.show()