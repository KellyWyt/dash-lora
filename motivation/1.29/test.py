import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
# 2. 数据定义 (人为设定)
# ==========================================
# 挑选 8 个代表性模块
# 格式要求: Layer X. type_{proj}
labels = [
    r'Layer 0. $q_{proj}$', 
    r'Layer 0. $k_{proj}$',   # k 较长
    r'Layer 10. $v_{proj}$',
    r'Layer 10. $up_{proj}$',  # up 较长
    r'Layer 20. $gate_{proj}$',
    r'Layer 20. $o_{proj}$',
    r'Layer 31. $up_{proj}$',  # up 较长
    r'Layer 31. $down_{proj}$' # down 最长
]

# 对应的 Sensitivity Score 数值
# 逻辑：up, down, k 普遍长一些 ( > 0.6 )，其他较短 ( < 0.4 )
values = [
    0.25, # q_proj (Short)
    0.65, # k_proj (Long)
    0.15, # v_proj (Short)
    0.75, # up_proj (Long)
    0.20, # gate_proj (Short)
    0.18, # o_proj (Short)
    0.82, # up_proj (Long)
    0.95  # down_proj (Very Long)
]

# 翻转列表，使得 Layer 0 在图表最上方
labels = labels[::-1]
values = values[::-1]

# ==========================================
# 3. 颜色映射逻辑
# ==========================================
# 使用蓝色渐变，数值越大颜色越深
cmap = cm.Blues
# 设置归一化范围 (0.0 - 1.0)
norm = mcolors.Normalize(vmin=0.1, vmax=1.0)
colors = [cmap(norm(v)) for v in values]

# ==========================================
# 4. 绘图逻辑 (水平条形图)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5)) # 调整长宽比以适应水平图

y_pos = np.arange(len(labels))

# 绘制水平柱子 (barh)
bars = ax.barh(y_pos, values, color=colors, edgecolor='#2b4f81', linewidth=0.8, height=0.6)

# ==========================================
# 5. 坐标轴与装饰
# ==========================================
# Y轴标签设置
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12, fontstyle='italic') # 名字倾斜着写(斜体)

# X轴设置
ax.set_xlabel(r'Sensitivity Score $\hat{n}_l$', fontsize=14)
ax.set_xlim(0, 1.1)

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 去掉左边框 (让文字悬浮)
ax.spines['left'].set_visible(False)

# 添加纵向网格线 (辅助看长度)
ax.grid(axis='x', linestyle='--', alpha=0.4)

# 添加阈值线 (垂直线)
threshold_x = 0.45
ax.axvline(x=threshold_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(threshold_x + 0.02, len(labels)-0.5, r'Threshold $\tau$', fontsize=11, color='gray', va='top')

# 添加解释性标注 (Annotate)
# 标注 Exclusive (长柱子)
ax.annotate('Exclusive Expert\n(High Sensitivity)', 
            xy=(0.95, 0), xycoords=('data', 'data'), # 指向最下面的 down_proj
            xytext=(0.6, 2.5), textcoords='data',
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"), 
            fontsize=10, ha='center')

# 标注 Shared (短柱子)
ax.annotate('Shared Substrate\n(Low Sensitivity)', 
            xy=(0.18, 2), xycoords=('data', 'data'), # 指向 o_proj
            xytext=(0.4, 4), textcoords='data',
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=-.2"), 
            fontsize=10, ha='center')

plt.tight_layout()

# ==========================================
# 6. 保存
# ==========================================
current_path = os.getcwd()
png_path = os.path.join(current_path, 'sensitivity_horizontal_bar.png')
# pdf_path = os.path.join(current_path, 'sensitivity_horizontal_bar.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight')
# plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

print(f"图片已保存至: {png_path}")
plt.show()