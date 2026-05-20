import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

#比较lora更新方向 和 更新力度

# ================= 配置路径 =================
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/fine_grained_analysis"

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TARGET_LAYERS = range(32)

def analyze_fine_grained(adapter_path):
    print(f"Loading weights from {adapter_path}...")
    adapter = torch.load(adapter_path, map_location='cpu')
    
    records = []
    
    print(">>> Scanning (Layer x Module) for Magnitude & Similarity...")
    for i in TARGET_LAYERS:
        for proj in TARGET_MODULES:
            # 构造 Key (根据你的文件结构微调)
            # 假设结构: ...layers.0.self_attn.q_proj.lora_A1.weight
            prefix = f"layers.{i}."
            proj_str = f".{proj}."
            
            keys = [k for k in adapter.keys() if prefix in k and proj_str in k]
            
            key_A1 = next((k for k in keys if "lora_A1" in k), None) # Video
            key_A2 = next((k for k in keys if "lora_A2" in k), None) # Audio
            
            if key_A1 and key_A2:
                t1 = adapter[key_A1].float().flatten()
                t2 = adapter[key_A2].float().flatten()
                
                # 1. 计算方向相似度 (Cosine Similarity)
                # 因为共享 B0，所以比较 A1 和 A2 的方向就等同于比较 Delta W 的方向
                sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
                
                # 2. 计算更新力度 (Magnitude)
                # 取两者的平均范数作为该模块的“活跃度”
                mag = (t1.norm() + t2.norm()).item() / 2.0
                
                records.append({
                    "Layer": i,
                    "Module": proj,
                    "Similarity": sim,
                    "Magnitude": mag
                })
    
    return pd.DataFrame(records)

def plot_bubble_chart(df):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    sns.set_theme(style="whitegrid")
    
    # 创建气泡图
    plt.figure(figsize=(16, 10))
    
    # 使用 seaborn scatterplot
    # hue: 颜色区分模块
    # size: 大小区分更新力度
    scatter = sns.scatterplot(
        data=df, 
        x="Layer", 
        y="Similarity", 
        hue="Module", 
        size="Magnitude",
        sizes=(20, 400), # 气泡最小和最大尺寸
        alpha=0.7,
        palette="tab10",
        edgecolor="black"
    )
    
    # 添加参考线
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, label="Orthogonal (0)")
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1, label="High Similarity (0.5)")
    
    plt.title("Fine-Grained Analysis: Direction vs. Magnitude\n(Size = Magnitude, Y-axis = Direction Similarity)", fontsize=16)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Cosine Similarity (Video vs. Audio)", fontsize=12)
    
    # 调整图例
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "fine_grained_bubble_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    
    # 保存原始数据方便查看
    df.to_csv(os.path.join(OUTPUT_DIR, "fine_grained_data.csv"), index=False)

if __name__ == "__main__":
    df = analyze_fine_grained(ADAPTER_BIN_PATH)
    plot_bubble_chart(df)