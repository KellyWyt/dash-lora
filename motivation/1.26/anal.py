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
modality_keys = {'Text': 'lora_A0', 'Video': 'lora_A1', 'Audio': 'lora_A2'}
# 假设是 Llama2 设置
LORA_RANK = 4 

# ================= 核心分析函数：不考虑 B 的 A 权重分析 =================
def analyze_pure_A_weights(path):
    print(f"Processing {path}...")
    if not os.path.exists(path): return None
    state_dict = torch.load(path, map_location="cpu")
    
    norm_text = 0.0
    norm_video = 0.0
    norm_audio = 0.0
    norm_initial = 0.0 # 初始化的 A 范数参考
    
    count = 0
    
    # 遍历 state_dict 寻找 A 矩阵
    # 逻辑：寻找包含 lora_A0 (Text) 的 key，并推导出同位置的 A1 (Video) 和 A2 (Audio)
    for key in state_dict.keys():
        if modality_keys['Text'] in key and ".weight" in key:
            prefix = key.split(modality_keys['Text'])[0]
            
            # 1. 提取 Text A
            A_t = state_dict[key].float()
            norm_text += torch.norm(A_t).item()
            
            # 2. 提取 Video A
            key_v = prefix + modality_keys['Video'] + ".weight"
            if key_v in state_dict:
                A_v = state_dict[key_v].float()
                norm_video += torch.norm(A_v).item()
                
                # 3. 计算“初始化状态”的范数参考
                # LoRA A 通常使用 Kaiming Uniform 初始化
                # 其范数的统计期望仅取决于矩阵形状
                A_init = torch.zeros_like(A_v)
                nn.init.kaiming_uniform_(A_init, a=np.sqrt(5))
                norm_initial += torch.norm(A_init).item()
            
            # 4. 提取 Audio A
            key_a = prefix + modality_keys['Audio'] + ".weight"
            if key_a in state_dict:
                A_a = state_dict[key_a].float()
                norm_audio += torch.norm(A_a).item()
                
            count += 1

    if count == 0: return None
    
    # 计算相对范数 (相对于 Text A)
    # 这反映了在没有 B 的情况下，专家 A 自身的更新强度
    rel_video = norm_video / norm_text
    rel_audio = norm_audio / norm_text
    rel_init = norm_initial / norm_text # 如果 Text A 练得非常强，这个值会变小
    
    return rel_video, rel_audio, rel_init

# ================= 执行计算 =================
results = {}
for ratio, path in checkpoint_paths.items():
    res = analyze_pure_A_weights(path)
    if res: results[ratio] = res

# ================= 绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

ratios = list(results.keys())
video_vals = [r[0] for r in results.values()]
init_vals = [r[2] for r in results.values()] # 初始权重能级比
x = np.arange(len(ratios))

# 1. 画 A 矩阵的相对强度柱状图
ax.bar(x, video_vals, color=['#ff9999', '#66b3ff', '#99e699'], 
       edgecolor='black', width=0.5, alpha=0.9, zorder=3, label='Pure Adapter A Norm (Video/Text)')

# 2. 画初始化基准线 (由于分母是练过的 Text A，所以这条线反映了 Text A 相比初始状态增强了多少)
ax.plot(x, init_vals, color='red', marker='o', linestyle='--', linewidth=2, 
        zorder=4, label='Initialization Base (Rel. to Trained Text)')

# 标注
ax.set_ylabel("Weight Norm Ratio (||Av|| / ||At||)")
ax.set_xlabel("Ratio Settings ($\Gamma$)")
ax.set_title("Intrinsic Adapter Weight Analysis (Excluding Shared B)", fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ratios)
ax.legend()

plt.tight_layout()
plt.savefig("pure_A_weight_analysis.png", dpi=300)
print("Saved pure_A_weight_analysis.png")