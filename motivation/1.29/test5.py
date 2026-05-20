import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import os

# ==========================================
# 1. 字体与全局设置
# ==========================================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300

# ==========================================
# 2. 数据定义 (保持你原始的 qproj 设置)
# ==========================================
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

# 模拟数值 (保持区分度)
values = [0.45, 0.75, 0.15, 0.82, 0.35, 0.25, 0.90, 0.98]

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
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=16)

# 横轴标题修改为 Energy Score，并保留数学符号一致性
ax.set_xlabel(r'Energy Score $\hat{n}_{l,\phi}$', fontsize=18)
ax.set_xlim(0, 1.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# 极简学术风格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

plt.tight_layout()

# ==========================================
# 6. 保存
# ==========================================
plt.savefig('energy_score_bar.pdf', dpi=300, bbox_inches='tight')
plt.show()