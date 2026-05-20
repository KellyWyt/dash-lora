import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ================= ⚙️ 配置路径 =================
ADAPTER_BIN_PATH = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
OUTPUT_DIR = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/cross_layer_all_modalities"

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TARGET_LAYERS = 32
MODALITIES = {
    'Text (A0)': 'lora_A0.weight',
    'Video (A1)': 'lora_A1.weight',
    'Audio (A2)': 'lora_A2.weight'
}

def get_layer_tensors(adapter, proj_type, modality_suffix):
    """提取指定模块、指定模态在所有 32 层中的 A 矩阵"""
    tensors = {}
    for i in range(TARGET_LAYERS):
        # 构造匹配模式
        # 匹配: layers.0. ... .proj_type.lora_A0.weight
        # 注意: 这里使用简单的字符串包含检查，适配你的 key 格式
        
        target_key = None
        for k in adapter.keys():
            # 必须同时包含: 层号, 模块名, 模态后缀
            # 且模块名必须精确匹配 (避免 up_proj 匹配到 gate_proj 如果命名有包含关系)
            if f"layers.{i}." in k and f".{proj_type}." in k and modality_suffix in k:
                target_key = k
                break
        
        if target_key:
            tensors[i] = adapter[target_key].float().flatten()
            
    return tensors

def compute_similarity_matrix(tensors):
    """计算 32x32 余弦相似度矩阵"""
    if not tensors:
        return None
    
    matrix = np.zeros((TARGET_LAYERS, TARGET_LAYERS))
    available_indices = sorted(tensors.keys())
    
    for i in range(TARGET_LAYERS):
        for j in range(TARGET_LAYERS):
            if i in tensors and j in tensors:
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j: # 计算上三角
                    sim = F.cosine_similarity(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0)).item()
                    matrix[i, j] = sim
                    matrix[j, i] = sim
            else:
                # 缺失层填充 NaN 或 0
                matrix[i, j] = 0.0
                
    return matrix

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"Loading Adapter: {ADAPTER_BIN_PATH}")
    try:
        adapter = torch.load(ADAPTER_BIN_PATH, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(">>> Starting Cross-Layer & Cross-Modality Analysis...")

    for proj in TARGET_MODULES:
        print(f"Analyzing {proj}...")
        
        # 创建画布：1行3列
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # 遍历三个模态
        for idx, (mod_name, suffix) in enumerate(MODALITIES.items()):
            ax = axes[idx]
            
            # 1. 提取数据
            tensors = get_layer_tensors(adapter, proj, suffix)
            
            # 2. 计算矩阵
            sim_matrix = compute_similarity_matrix(tensors)
            
            # 3. 绘图
            if sim_matrix is not None:
                # 使用统一的色阶，方便横向对比 (-0.2 到 0.8)
                sns.heatmap(sim_matrix, ax=ax, cmap="coolwarm", vmin=-0.2, vmax=0.8, 
                            cbar=True, square=True)
                ax.set_title(f"{mod_name}\nCross-Layer Similarity", fontsize=14)
                ax.set_xlabel("Layer Index")
                ax.set_ylabel("Layer Index")
            else:
                ax.text(0.5, 0.5, "No Data Found", ha='center', va='center')
                ax.set_title(f"{mod_name} (Missing)")

        plt.suptitle(f"Module: {proj}", fontsize=18, y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"heatmap_all_modalities_{proj}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")

    print("\n>>> Analysis Complete!")

if __name__ == "__main__":
    main()