import torch
import torch.nn.functional as F
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

modality_keys = {'Text': 'lora_A0', 'Video': 'lora_A1', 'Audio': 'lora_A2'}
shared_b_key = 'lora_B0'

# ================= 计算冲突度 =================
def analyze_conflict(path):
    print(f"Analyzing {path}...")
    if not os.path.exists(path): return None
    state_dict = torch.load(path, map_location="cpu")
    
    similarities = [] # 存储每一层的 Text vs Video 余弦相似度
    
    for key in state_dict.keys():
        if shared_b_key in key:
            prefix = key.split(shared_b_key)[0]
            # 获取 Text A 和 Video A
            key_text = prefix + modality_keys['Text'] + ".weight"
            key_video = prefix + modality_keys['Video'] + ".weight"
            
            if key_text in state_dict and key_video in state_dict:
                # 为了计算方便，我们将矩阵展平
                vec_text = state_dict[key_text].float().view(-1)
                vec_video = state_dict[key_video].float().view(-1)
                
                # 计算余弦相似度: (A . B) / (|A| |B|)
                sim = F.cosine_similarity(vec_text.unsqueeze(0), vec_video.unsqueeze(0))
                similarities.append(sim.item())
                
    return np.mean(similarities), np.std(similarities)

# ================= 执行 =================
results = {}
for ratio, path in checkpoint_paths.items():
    res = analyze_conflict(path)
    if res: results[ratio] = res

# ================= 绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
fig, ax = plt.subplots(figsize=(8, 6))

ratios = list(results.keys())
means = [results[r][0] for r in ratios]
stds = [results[r][1] for r in ratios]

# 我们原本期望协同，如果相似度太低甚至接近0，说明正交/冲突严重
# 这里我们定义 "Conflict Score" = 1 - Abs(Similarity) 或者直接看 Similarity
# 如果 2:1:1 的相似度 显著低于 3:1:1，说明 2:1:1 中视频在乱指方向

ax.bar(ratios, means, yerr=stds, capsize=5, color=['#ff9999','#66b3ff','#99ff99'], alpha=0.8, edgecolor='black')
ax.set_ylabel("Text-Video Directional Alignment (Cosine Sim)")
ax.set_title("Directional Conflict Analysis")
plt.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig("semantic_conflict_analysis.png", dpi=300)
print("Done. Saved semantic_conflict_analysis.png")