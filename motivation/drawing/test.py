# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os

# def draw_dash_lora_refined():
#     fig, ax = plt.subplots(figsize=(14, 8))
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0, 60)
#     ax.axis('off')

#     # --- Stage 1: Intrinsic Sensitivity Profiling (左侧：分色模块层级) ---
#     ax.text(2, 55, "Stage 1: Sensitivity Profiling", fontsize=12, fontweight='bold')
    
#     # 模拟从 Layer 1 到 Layer n 的演变
#     # 颜色梯度映射：深红色代表高敏感度(Expert)，浅灰色代表低敏感度(Shared)
#     layer_configs = [
#         {"name": "Layer 1", "color": "#FF4D4D", "text_col": "white", "label": "High"},
#         {"name": "Layer 2", "color": "#FFCCCC", "text_col": "black", "label": "Low"},
#         {"name": "...",     "color": "white",   "text_col": "black", "label": ""}, # 省略号层
#         {"name": "Layer i", "color": "#FF8080", "text_col": "white", "label": "Med"},
#         {"name": "Layer n", "color": "#F2F2F2", "text_col": "black", "label": "Min"}
#     ]

#     for i, config in enumerate(reversed(layer_configs)):
#         y_pos = 10 + i * 8
#         is_dots = config["name"] == "..."
        
#         # 绘制模块方块
#         rect = patches.Rectangle((4, y_pos), 18, 6, linewidth=1.5, 
#                                  edgecolor='black' if not is_dots else 'none', 
#                                  facecolor=config["color"])
#         ax.add_patch(rect)
        
#         # 居中放置文字，确保对比度
#         ax.text(13, y_pos + 2.2, config["name"], fontsize=10, 
#                 fontweight='bold', color=config["text_col"], ha='center')
        
#         # 标注敏感度量级 (如 S_l 数值)
#         if not is_dots:
#             ax.text(23, y_pos + 2.2, f"$S_l$ {config['label']}", fontsize=9, style='italic', color='#555555')

#     ax.text(2, 5, r"Atomic-level Mapping: $\mathcal{F} \rightarrow \{S_l\}$", fontsize=10, fontweight='bold')

#     # --- 后续阶段逻辑保持 (Stage 2 & 3) ---
#     # 连接箭头 (从 Layer 集群指向调度器)
#     ax.annotate('', xy=(35, 30), xytext=(22, 30), arrowprops=dict(arrowstyle='->', lw=2))

#     # Stage 2: Funnel (省略细节代码以节省篇幅，保持之前逻辑)
#     funnel_pts = [[38, 45], [52, 45], [47, 20], [43, 20]]
#     ax.add_patch(patches.Polygon(funnel_pts, closed=True, lw=2, edgecolor='black', facecolor='#F8F8F8'))
#     ax.text(41, 32, r"Budget $K$", fontsize=11, fontweight='bold')

#     # Stage 3: Asymmetric Topology (3:1:1 核心展示)
#     ax.add_patch(patches.Rectangle((62, 33), 36, 20, lw=1, edgecolor='#FFA500', facecolor='#FFFFF5', ls='--'))
#     # Text Anchor (3层)
#     for j in range(3):
#         ax.add_patch(patches.Rectangle((67, 36 + j*3.5), 9, 3, facecolor='#FFD700', edgecolor='black'))
#     ax.text(67, 48, "Text Anchor", fontsize=9, fontweight='bold')
#     ax.text(78, 41, r"$\gamma_t=3$", fontsize=11, fontweight='bold')
#     # V/A (1层)
#     ax.add_patch(patches.Rectangle((87, 41), 9, 3, facecolor='#87CEEB', edgecolor='black'))
#     ax.add_patch(patches.Rectangle((87, 36), 9, 3, facecolor='#98FB98', edgecolor='black'))
#     ax.text(97, 41, r"$\gamma_p=1$", fontsize=11)

#     # 共享池
#     pool = patches.Circle((80, 15), 5, facecolor='#EAEAEA', edgecolor='black')
#     ax.add_patch(pool)
#     ax.text(76, 14, r"Pool $\mathcal{G}$", fontsize=11, fontweight='bold')

#     plt.title("DASH-LoRA: From Intrinsic Energy to Asymmetric Topology", fontsize=16, pad=25)
#     save_path = "dash_lora_v2_color_coded.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Diagram saved to: {os.path.abspath(save_path)}")

# if __name__ == "__main__":
#     draw_dash_lora_refined()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def draw_radar_dash_lora():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # --- Stage 1: Sensitivity Profiling with Atomic Radar Charts ---
    ax.text(2, 55, "Stage 1: Sensitivity Profiling", fontsize=12, fontweight='bold')
    
    # 模拟层级结构 (Layer 1 ... Layer n)
    layers = [
        {"name": "Layer 1", "pos": 45, "data": [0.9, 0.8, 0.4, 0.7, 0.8], "color": "#FF4D4D"}, # 高能
        {"name": "Layer 2", "pos": 37, "data": [0.6, 0.5, 0.3, 0.4, 0.5], "color": "#FFCCCC"},
        {"name": "...",     "pos": 29, "data": None, "color": "white"},
        {"name": "Layer i", "pos": 21, "data": [0.4, 0.3, 0.2, 0.3, 0.4], "color": "#FF8080"},
        {"name": "Layer n", "pos": 13, "data": [0.2, 0.1, 0.1, 0.1, 0.2], "color": "#F2F2F2"}  # 低能
    ]

    for layer in layers:
        y = layer["pos"]
        is_dots = layer["name"] == "..."
        # 绘制基础方块
        rect = patches.Rectangle((4, y), 18, 6, lw=1.5, ec='black' if not is_dots else 'none', fc=layer["color"])
        ax.add_patch(rect)
        ax.text(13, y + 2.2, layer["name"], fontsize=10, fontweight='bold', ha='center')

        # 只为 Layer 1 和 Layer n 画雷达图
        if layer["name"] in ["Layer 1", "Layer n"]:
            # 雷达图中心坐标
            center_x, center_y = 32, y + 3
            num_vars = 5
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            # 模拟投影维度: q, k, v, o, gate
            data = layer["data"] + [layer["data"][0]]
            scale = 4  # 雷达图缩放倍数
            
            # 绘制雷达图轴线
            for angle in angles[:-1]:
                ax.plot([center_x, center_x + np.cos(angle)*scale], 
                        [center_y, center_y + np.sin(angle)*scale], color='gray', lw=0.5, alpha=0.5)
            
            # 绘制雷达图填充区域
            values_x = [center_x + np.cos(a) * v * scale for a, v in zip(angles, data)]
            values_y = [center_y + np.sin(a) * v * scale for a, v in zip(angles, data)]
            ax.fill(values_x, values_y, color=layer["color"], alpha=0.6)
            ax.plot(values_x, values_y, color='red' if layer["name"]=="Layer 1" else 'gray', lw=1)
            
            # 标注雷达图含义
            label_pos = 50 if layer["name"] == "Layer 1" else 10
            ax.text(center_x, y + 8, "Atomic Sensitivity", fontsize=8, ha='center', style='italic')

    # --- 后续 Stage 2 & 3 保持逻辑 ---
    ax.annotate('', xy=(45, 30), xytext=(24, 30), arrowprops=dict(arrowstyle='->', lw=2))
    
    # 简化 Stage 2 (漏斗)
    funnel = patches.Polygon([[48, 45], [62, 45], [57, 20], [53, 20]], closed=True, fc='#F8F8F8', ec='black', lw=2)
    ax.add_patch(funnel)
    ax.text(51, 32, r"Budget $K$", fontsize=11, fontweight='bold')

    # 简化 Stage 3 (3:1:1 结构)
    ax.add_patch(patches.Rectangle((70, 35), 25, 18, lw=1, ec='#FFA500', fc='#FFFFF5', ls='--'))
    for j in range(3): # Text Anchor
        ax.add_patch(patches.Rectangle((74, 38+j*3), 6, 2.5, fc='#FFD700', ec='black'))
    ax.text(82, 42, r"$\gamma_t=3$", fontsize=12, fontweight='bold')
    
    plt.title("DASH-LoRA: Atomic Sensitivity Profiling & Asymmetric Topology", fontsize=14, pad=20)
    save_path = "dash_lora_radar_v1.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    draw_radar_dash_lora()