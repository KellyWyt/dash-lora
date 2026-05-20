import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 配置 =================
bin_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"
modalities = ['Text', 'Video', 'Audio']
modality_keys = ['lora_A0', 'lora_A1', 'lora_A2'] 
shared_b_key = 'lora_B0'
num_layers = 32

# ================= 数据加载 =================
print(f"Loading weights from {bin_path}...")
state_dict = torch.load(bin_path, map_location="cpu")

# 收集每个模态的所有模块能量（不分层，混在一起）
all_module_energies = {m: [] for m in modalities}

print("Profiling energy magnitude...")
for i in range(num_layers):
    layer_pattern = f"layers.{i}."
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
                # 计算能量
                delta_W = torch.matmul(B, A)
                energy = torch.norm(delta_W).item()
                all_module_energies[m_name].append(energy)

# ================= 洛伦兹曲线计算 (Lorenz Analysis) =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
colors = ['#FFD700', '#87CEEB', '#98FB98'] # 黄 蓝 绿

# 设定的“有效能量阈值” (Gini Threshold)
THRESHOLD = 0.90 

for idx, m in enumerate(modalities):
    energies = np.array(all_module_energies[m])
    
    # 关键步骤：按能量从大到小排序
    sorted_energies = np.sort(energies)[::-1]
    
    # 计算累积和
    cumsum = np.cumsum(sorted_energies)
    total_energy = cumsum[-1] + 1e-6
    normalized_cumsum = cumsum / total_energy * 100 # y轴: 累积能量百分比
    
    # 生成 x轴: 涉及的模块百分比 (Top-K %)
    x_percent = np.arange(1, len(energies) + 1) / len(energies) * 100
    
    # 绘图
    plt.plot(x_percent, normalized_cumsum, label=f"{m} (Gini)", color=colors[idx], linewidth=3)
    
    # 找到达到 90% 能量所需的模块比例
    cutoff_idx = np.argmax(normalized_cumsum >= THRESHOLD * 100)
    cutoff_x = x_percent[cutoff_idx]
    
    # 标记这一关键点
    plt.scatter([cutoff_x], [normalized_cumsum[cutoff_idx]], color=colors[idx], s=100, zorder=5, edgecolors='k')
    plt.text(cutoff_x + 2, normalized_cumsum[cutoff_idx] - 5, f"{m}\nTop {cutoff_x:.1f}%", fontsize=11, fontweight='bold')
    
    # 画垂直线
    plt.vlines(cutoff_x, 0, normalized_cumsum[cutoff_idx], colors=colors[idx], linestyles=':', alpha=0.6)

# 装饰图表
plt.axhline(y=90, color='gray', linestyle='--', alpha=0.8, label='90% Information Retention')
plt.title("Sparsity Analysis: Energy Lorenz Curve", fontsize=14, fontweight='bold')
plt.xlabel("Percentage of Parameters Used (Top-K %)")
plt.ylabel("Cumulative Energy Captured (%)")
plt.xlim(0, 100)
plt.ylim(0, 105)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("sparsity_lorenz_curve.png", dpi=300)
plt.show()