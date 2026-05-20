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
# 标签格式修改: $l_{下标} : 变量_{下标}$
# 例如: r'$l_0 : q_{proj}$'
labels = [
    r'$l_0 : q_{proj}$', 
    r'$l_0 : k_{proj}$',   
    r'$l_{10} : v_{proj}$',
    r'$l_{10} : up_{proj}$',  
    r'$l_{20} : gate_{proj}$',
    r'$l_{20} : o_{proj}$',
    r'$l_n : up_{proj}$',  
    r'$l_n : down_{proj}$' 
]

# 数值 (保持之前优化过的区分度)
values = [
    0.45, # q_proj (中等浅)
    0.75, # k_proj (深)
    0.15, # v_proj (最浅)
    0.82, # up_proj (深)
    0.35, # gate_proj (浅)
    0.25, # o_proj (更浅)
    0.90, # up_proj (更深)
    0.98  # down_proj (最深)
]

# 翻转数据，使 layer 0 在图表最上方
labels = labels[::-1]
values = values[::-1]

# ==========================================
# 3. 颜色映射逻辑
# ==========================================
cmap = cm.Blues
norm = mcolors.Normalize(vmin=0.05, vmax=1.0)
colors = [cmap(norm(v)) for v in values]

# ==========================================
# 4. 绘图逻辑
# ==========================================
fig, ax = plt.subplots(figsize=(5, 8))

y_pos = np.arange(len(labels))

# 绘制水平柱子
bars = ax.barh(y_pos, values, color=colors, edgecolor='#2b4f81', linewidth=0.8, height=0.6)

# ==========================================
# 5. 坐标轴与装饰
# ==========================================
# Y轴标签设置 (LaTeX格式会自动渲染下标)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=16)

# X轴设置
ax.set_xlabel(r'Sensitivity Score $\hat{n}_l$', fontsize=16)
ax.set_xlim(0, 1.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# 去掉边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# 确保无网格
ax.grid(False)

# 紧凑布局
plt.tight_layout()

# ==========================================
# 6. 保存
# ==========================================
current_path = os.getcwd()
png_path = os.path.join(current_path, 'sensitivity_bar_colon.png')
pdf_path = os.path.join(current_path, 'sensitivity_bar_colon.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

print(f"图片已保存至: {png_path}")
plt.show()