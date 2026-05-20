import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import os

# ==========================================
# 1. 字体与全局设置
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
# 标签
labels = [
    r'layer 0. $q_{proj}$', 
    r'layer 0. $k_{proj}$',
    r'layer 10. $v_{proj}$',
    r'layer 19. $up_{proj}$',
    r'layer 28. $gate_{proj}$',
    r'layer 21. $o_{proj}$',
    r'layer n. $up_{proj}$',
    r'layer n. $down_{proj}$'
]

# 数值 (保持之前的设定以维持区分度)
values = [0.38, 0.72, 0.12, 0.78, 0.28, 0.22, 0.88, 0.98]

# 翻转数据
labels = labels[::-1]
values = values[::-1]

# ==========================================
# 3. 颜色映射逻辑
# ==========================================
cmap = cm.Blues
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
colors = [cmap(norm(v)) for v in values]

# ==========================================
# 4. 绘图逻辑 (竖长矩形)
# ==========================================
fig, ax = plt.subplots(figsize=(5, 8))
y_pos = np.arange(len(labels))

# 绘制水平柱子
bars = ax.barh(y_pos, values, color=colors, edgecolor='#2b4f81', linewidth=0.8, height=0.6)

# ==========================================
# 5. 坐标轴与装饰
# ==========================================
# Y轴标签
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=14, fontstyle='italic')

# X轴设置 (密集刻度)
ax.set_xlabel(r'Sensitivity Score $\hat{n}_l$', fontsize=16)
ax.set_xlim(0, 1.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# 去掉上、右、左边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# 【核心修改】去掉纵向网格线
# ax.grid(axis='x', which='major', linestyle='--', alpha=0.5)
# ax.grid(axis='x', which='minor', linestyle=':', alpha=0.3)

# 紧凑布局
plt.tight_layout()

# ==========================================
# 6. 保存
# ==========================================
current_path = os.getcwd()
png_path = os.path.join(current_path, 'sensitivity_bar_vertical_clean.png')
pdf_path = os.path.join(current_path, 'sensitivity_bar_vertical_clean.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

print(f"图片已保存至: {png_path}")
plt.show()