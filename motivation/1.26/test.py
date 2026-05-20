import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 配置区域 =================
bin_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"
num_layers = 32
modalities = ['Text', 'Video', 'Audio']
modality_keys = ['lora_A0', 'lora_A1', 'lora_A2']
shared_b_key = 'lora_B0'

# 模拟的全局瓶颈 sigma (比如 0.22 意味着总共有多少个 exclusive slots)
# 这里我们假设 total_modules = num_layers * 7 (projection types) * 3 (modalities)
# 为了简化演示，我们只统计 adapter_model.bin 里实际存在的模块
# ===========================================

print(f"Loading weights from {bin_path}...")
state_dict = torch.load(bin_path, map_location="cpu")

# 1. 提取所有模块的能量 (Frobenius Norm)
# Structure: module_energy_list = [{'layer': i, 'modality': 'Text', 'energy': float, 'name': '...'}, ...]
module_energies = []

print("Calculating energy for all modules...")
for i in range(num_layers):
    layer_pattern = f"layers.{i}."
    # 找到该层所有的 projecting modules (q, k, v, o, gate, up, down)
    layer_keys = [k for k in state_dict.keys() if layer_pattern in k and shared_b_key in k]
    
    # 提取模块前缀 (e.g., layers.0.self_attn.q_proj.)
    prefixes = set([k.split(shared_b_key)[0] for k in layer_keys])
    
    for prefix in prefixes:
        key_B = prefix + f"{shared_b_key}.weight"
        if key_B not in state_dict: continue
        B = state_dict[key_B].float()
        
        for m_idx, m_name in enumerate(modalities):
            key_A = prefix + f"{modality_keys[m_idx]}.weight"
            if key_A in state_dict:
                A = state_dict[key_A].float()
                # 计算 ΔW = B @ A 的能量
                delta_W = torch.matmul(B, A)
                energy = torch.norm(delta_W).item()
                
                module_energies.append({
                    'layer': i,
                    'modality': m_name,
                    'energy': energy,
                    'id': f"{prefix}{m_name}"
                })

# 2. 模拟不同比例下的能量捕获率
# 定义三种实验设置: {Name: (Text_ratio, Video_ratio, Audio_ratio)}
ratios_settings = {
    '2:1:1': (2, 1, 1),
    '3:1:1 (Ours)': (3, 1, 1),
    '4:1:1': (4, 1, 1)
}

# 假设总预算 K 是固定的 (基于 sigma=0.22)
total_modules_count = len(module_energies)
# 假设 sigma = 0.22，即保留 22% 的模块为 Expert
sigma = 0.22 
total_budget_K = int(total_modules_count * sigma)

print(f"Total Modules: {total_modules_count}")
print(f"Total Exclusive Budget (K): {total_budget_K} (sigma={sigma})")

results = {} # Store captured energy percentages

for setting_name, ratios in ratios_settings.items():
    # 计算各模态的配额
    ratio_sum = sum(ratios)
    quota_text = int(total_budget_K * (ratios[0] / ratio_sum))
    quota_video = int(total_budget_K * (ratios[1] / ratio_sum))
    quota_audio = int(total_budget_K * (ratios[2] / ratio_sum))
    
    # 分模态排序并截断
    captured_energy = {m: 0.0 for m in modalities}
    total_energy = {m: 0.0 for m in modalities}
    
    for m in modalities:
        # 获取该模态所有模块并按能量降序排列
        m_modules = [x for x in module_energies if x['modality'] == m]
        m_modules.sort(key=lambda x: x['energy'], reverse=True)
        
        # 计算该模态的总能量
        m_total_energy = sum([x['energy'] for x in m_modules])
        total_energy[m] = m_total_energy
        
        # 确定截断阈值
        if m == 'Text': quota = quota_text
        elif m == 'Video': quota = quota_video
        else: quota = quota_audio
        
        # 获取 Top-K 模块的能量之和
        top_k_modules = m_modules[:quota]
        m_captured = sum([x['energy'] for x in top_k_modules])
        captured_energy[m] = m_captured
        
    # 计算捕获百分比
    results[setting_name] = {m: (captured_energy[m] / total_energy[m] * 100) for m in modalities}

# --- Visualization ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

# 准备数据绘图
labels = list(results.keys()) # ['2:1:1', '3:1:1', '4:1:1']
text_vals = [results[k]['Text'] for k in labels]
video_vals = [results[k]['Video'] for k in labels]
audio_vals = [results[k]['Audio'] for k in labels]

x = np.arange(len(labels))
width = 0.25

rects1 = ax.bar(x - width, text_vals, width, label='Text (Anchor)', color='#FFD700', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, video_vals, width, label='Video', color='#87CEEB', edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + width, audio_vals, width, label='Audio', color='#98FB98', edgecolor='black', alpha=0.9)

# 添加阈值线 (假设 80% 能量捕获是及格线)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Critical Energy Threshold')

ax.set_ylabel('Energy Coverage (%)')
ax.set_title('Impact of Asymmetric Ratios on Energy Coverage')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 110)
ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
save_path = "ratio_analysis_coverage.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Analysis complete. Plot saved to {save_path}")
plt.show()