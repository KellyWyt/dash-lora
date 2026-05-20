import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 配置 =================
bin_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"
num_layers = 32
modalities = ['Text', 'Video', 'Audio']
modality_keys = ['lora_A0', 'lora_A1', 'lora_A2'] # 对应 Text, Video, Audio
shared_b_key = 'lora_B0'

# ================= 加载与计算 =================
print(f"Loading weights from {bin_path}...")
state_dict = torch.load(bin_path, map_location="cpu")

# 存储每一层的能量 (Norm)
layer_energies = {m: np.zeros(num_layers) for m in modalities}

print("Calculating layer-wise energy...")
for i in range(num_layers):
    layer_pattern = f"layers.{i}."
    
    # 找到该层所有相关的 key
    layer_keys = [k for k in state_dict.keys() if layer_pattern in k and shared_b_key in k]
    prefixes = set([k.split(shared_b_key)[0] for k in layer_keys])
    
    for prefix in prefixes:
        key_B = prefix + f"{shared_b_key}.weight"
        if key_B not in state_dict: continue
        B = state_dict[key_B].float()
        
        for m_idx, m_name in enumerate(modalities):
            key_A = prefix + f"{modality_keys[m_idx]}.weight"
            if key_A in state_dict:
                A = state_dict[key_A].float()
                # 计算 ΔW = B @ A 的能量 (Frobenius Norm)
                delta_W = torch.matmul(B, A)
                energy = torch.norm(delta_W).item()
                
                # 累加该层该模态的总能量
                layer_energies[m_name][i] += energy

# ================= 累积能量分析 (关键步骤) =================
cumulative_ratios = {m: np.zeros(num_layers) for m in modalities}
saturation_points = {} # 记录达到 90% 能量的层数
THRESHOLD = 0.90       # 设定有效能量阈值为 90%

for m in modalities:
    raw = layer_energies[m]
    total = np.sum(raw) + 1e-6
    # 计算累积分布 (CDF)
    cumsum = np.cumsum(raw)
    normalized_cumsum = cumsum / total
    cumulative_ratios[m] = normalized_cumsum
    
    # 找到第一个超过阈值的层
    sat_idx = np.argmax(normalized_cumsum >= THRESHOLD)
    saturation_points[m] = sat_idx

print("Saturation Points (90% Energy):", saturation_points)

# ================= 学术绘图 =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

plt.figure(figsize=(10, 6))
colors = ['#FFD700', '#87CEEB', '#98FB98'] # 黄(Text), 蓝(Video), 绿(Audio)
lines = []

x = np.arange(num_layers)

for idx, m in enumerate(modalities):
    y = cumulative_ratios[m] * 100 # 转为百分比
    line, = plt.plot(x, y, label=f"{m} (CDF)", color=colors[idx], linewidth=3)
    lines.append(line)
    
    # 标记饱和点
    sat_layer = saturation_points[m]
    sat_val = y[sat_layer]
    plt.scatter([sat_layer], [sat_val], color=colors[idx], s=100, zorder=5, edgecolors='k')
    
    # 画垂直虚线指示饱和层数
    plt.vlines(sat_layer, 0, sat_val, colors=colors[idx], linestyles=':', alpha=0.8)
    plt.text(sat_layer + 0.5, 10 + idx*5, f"{m}\nSat @ L{sat_layer}", color='black', fontsize=10)

# 画阈值线
plt.axhline(y=THRESHOLD*100, color='gray', linestyle='--', alpha=0.6, label='90% Effective Threshold')

plt.title("Cumulative Adaptation Energy Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Layer Depth (0-31)")
plt.ylabel("Cumulative Energy (%)")
plt.xlim(0, 31)
plt.ylim(0, 105)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("cumulative_energy_saturation.png", dpi=300)
plt.show()