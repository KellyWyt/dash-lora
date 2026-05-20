import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 1. 字体与样式设置 (ICML/学术风格)
# ==========================================
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
except:
    pass
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300

# ==========================================
# 2. 数据定义
# ==========================================
layers = ['Layer 0', 'Layer 1', 'Layer n']
proj_types = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
x = np.arange(len(proj_types))

# Sensitivity Scores
y_l0 = [0.12, 0.35, 0.15, 0.10, 0.12, 0.11, 0.20]
y_l1 = [0.20, 0.18, 0.22, 0.19, 0.25, 0.65, 0.21]
y_ln = [0.30, 0.28, 0.32, 0.35, 0.31, 0.33, 0.95]

# ==========================================
# 3. 颜色映射逻辑 (核心修改)
# ==========================================
# 定义色系：使用蓝色渐变 (Blues)
cmap = cm.Blues

# 定义颜色归一化：将 0~1.0 的数值映射到颜色
# vmin=-0.2 是为了让数值为0.1的柱子也有颜色，不至于变成纯白
norm = mcolors.Normalize(vmin=-0.1, vmax=1.1)

# 辅助函数：根据数值列表返回对应的颜色列表
def get_colors(values):
    return [cmap(norm(v)) for v in values]

# ==========================================
# 4. 绘图逻辑
# ==========================================
fig, ax = plt.subplots(figsize=(12, 5))

width = 0.25
offset = 8

# 绘制 Layer 0 (所有柱子都有颜色，基于数值)
bars0 = ax.bar(x, y_l0, width, label='Layer 0', 
               color=get_colors(y_l0), edgecolor='#5B84B1', linewidth=0.8)

# 绘制 Layer 1
bars1 = ax.bar(x + offset, y_l1, width, label='Layer 1', 
               color=get_colors(y_l1), edgecolor='#5B84B1', linewidth=0.8)

# 绘制 Layer n
barsn = ax.bar(x + offset * 2.2, y_ln, width, label='Layer n', 
               color=get_colors(y_ln), edgecolor='#5B84B1', linewidth=0.8)

# ==========================================
# 5. 坐标轴与装饰
# ==========================================
all_ticks = np.concatenate([x, x + offset, x + offset * 2.2])
all_labels = proj_types * 3
ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels, rotation=0, fontsize=10)

# Layer 分组标签
trans = ax.get_xaxis_transform()
ax.text(3, -0.12, 'Layer 0', ha='center', fontsize=12, fontweight='bold', transform=trans)
ax.text(3 + offset, -0.12, 'Layer 1', ha='center', fontsize=12, fontweight='bold', transform=trans)
ax.text(3 + offset * 2.2, -0.12, 'Layer n', ha='center', fontsize=12, fontweight='bold', transform=trans)

# 省略号
ax.text(3 + offset * 1.6, 0.5, '......', ha='center', fontsize=24, fontweight='bold', color='#265e9d')

# Y轴
ax.set_ylabel(r'Sensitivity Score $\hat{n}_l$', fontsize=14)
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 阈值线
ax.axhline(y=0.4, color='#5B84B1', linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(offset * 2.8, 0.42, r'Sharing Threshold $\tau$', fontsize=11, color='#5B84B1', ha='right', style='italic')

# 箭头标注 (文本颜色统一为深色，保持学术感)
ax.annotate('Shared Substrate\n(Low Sensitivity)', xy=(1, 0.35), xytext=(2.5, 0.65),
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"), fontsize=10)
ax.annotate('Exclusive Expert\n(High Sensitivity)', xy=(6 + offset * 2.2, 0.95), xytext=(2 + offset * 2.2, 0.95),
            arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

plt.tight_layout()

# ==========================================
# 6. 保存
# ==========================================
current_path = os.getcwd()
png_path = os.path.join(current_path, 'sensitivity_score_unified.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {png_path}")

plt.show()