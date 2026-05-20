import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os

# ================= 配置路径 =================
# 修改为你的 bin 文件实际路径
bin_file_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"

# 假设模态映射关系（你可以根据实际情况修改标签）
# 例如: 0: Text, 1: Image, 2: Audio
MODALITY_MAP = {
    0: "Text",
    1: "Video",
    2: "Audio"
}

def analyze_multimodal_lora(file_path):
    print(f"Loading weights from {file_path}...")
    try:
        state_dict = torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # 数据结构: data_map[layer][module][proj] = {'A': {0:..., 1:...}, 'B': {0:...}}
    data_map = {}
    
    # 正则解析
    # 匹配: layers.0.self_attn.q_proj.lora_A0.weight
    # Group 1: Layer Index
    # Group 2: Module (self_attn / mlp)
    # Group 3: Proj Type (q_proj, gate_proj...)
    # Group 4: Matrix Type (A/B)
    # Group 5: Modality Index (0, 1, 2...)
    pattern = re.compile(r"layers\.(\d+)\.(.+?)\.(.+?)\.lora_([AB])(\d+)\.weight")

    print("Parsing weights structure...")
    for key, tensor in state_dict.items():
        match = pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            module_name = match.group(2)
            proj_name = match.group(3)
            matrix_type = match.group(4)
            modality_idx = int(match.group(5))

            if layer_idx not in data_map: data_map[layer_idx] = {}
            if module_name not in data_map[layer_idx]: data_map[layer_idx][module_name] = {}
            if proj_name not in data_map[layer_idx][module_name]: 
                data_map[layer_idx][module_name][proj_name] = {'A': {}, 'B': {}}
            
            data_map[layer_idx][module_name][proj_name][matrix_type][modality_idx] = tensor.float()

    print(f"Parsed {len(data_map)} layers.")

    results = []

    print("Computing Magnitude (Frobenius Norm) per Modality...")
    for layer_idx in tqdm(sorted(data_map.keys())):
        for module in data_map[layer_idx]:
            for proj in data_map[layer_idx][module]:
                matrices = data_map[layer_idx][module][proj]
                
                # 获取共享的 B0 (假设只有一个 B0 用于投影回原空间)
                # 如果是 MoE 且 B 也是分开的，这里代码会自动适配取对应的 B
                bs = matrices['B']
                as_dict = matrices['A']
                
                if not as_dict: continue

                # 遍历每一个模态 A0, A1, A2
                for mod_idx, a_tensor in as_dict.items():
                    # 尝试寻找对应的 B。通常只有 B0。
                    # 如果有 B0, B1, B2，则一一对应；如果只有 B0，则共享。
                    b_tensor = bs.get(mod_idx)
                    if b_tensor is None:
                        b_tensor = bs.get(0) # Fallback to shared B0
                    
                    if b_tensor is None: continue # Skip if no B matrix found

                    # 计算 Delta W = B @ A
                    # 注意：A 的 shape [r, dim], B 的 shape [dim, r]
                    # Delta W shape [dim, dim]
                    
                    # 为了效率，我们利用 Trace 性质或直接计算 Norm
                    # Norm(B@A)
                    delta_w = torch.matmul(b_tensor, a_tensor)
                    norm_val = torch.norm(delta_w, p='fro').item()
                    
                    # 记录详细信息
                    results.append({
                        "Layer": layer_idx,
                        "Module_Type": module, # self_attn vs mlp
                        "Proj_Type": proj,     # q_proj, up_proj...
                        "Modality_ID": mod_idx,
                        "Modality_Label": MODALITY_MAP.get(mod_idx, f"Modality_{mod_idx}"),
                        "Magnitude": norm_val
                    })

    return pd.DataFrame(results)

def plot_multimodal_analysis(df):
    if df is None or df.empty:
        print("No data to plot.")
        return
    
    sns.set_theme(style="whitegrid")

    # # ================= 实验 1: 不同模态在不同层级的活跃度对比 =================
    # # 目的：回答 "是否某个模态在深层就“死”了？"
    # # 按 Layer 和 Modality 聚合平均 Magnitude
    # print("Plotting Layer-wise Modality Trends...")
    # plt.figure(figsize=(14, 6))
    # sns.lineplot(data=df, x="Layer", y="Magnitude", hue="Modality_Label", marker="o", linewidth=2.5)
    # plt.title("Exp 1: Layer-wise Update Magnitude by Modality (Where is the redundancy?)", fontsize=14)
    # plt.ylabel("Avg Delta W Norm (Update Strength)")
    # plt.xlabel("Layer Index")
    # plt.legend(title="Modality")
    # plt.savefig("exp1_layer_modality_trend.png")
    # plt.close()

    # ================= 实验 2: 投影类型(Proj)与模态的偏好分析 =================
    # 目的：回答 "Attention 对图像更重要，还是 MLP 对文本更重要？"
    # 排除 Layer 因素，看整体 Proj 类型的分布
    print("Plotting Projector Preferences...")
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="Proj_Type", y="Magnitude", hue="Modality_Label", errorbar=None)
    # plt.title("Exp 2: Which Modules does each Modality prefer?", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("Avg Magnitude")
    plt.savefig("exp2_proj_modality_pref.pdf")
    plt.close()

    # # ================= 实验 3: 细粒度热力图 (分模态) =================
    # # 目的：精确定位冗余。例如 "Modality 2 在 Layer 20-30 的 MLP 上全是蓝色(0)"
    # print("Plotting Heatmaps...")
    
    # unique_modalities = sorted(df['Modality_ID'].unique())
    # for mod_id in unique_modalities:
    #     mod_label = MODALITY_MAP.get(mod_id, f"Mod_{mod_id}")
    #     subset = df[df['Modality_ID'] == mod_id]
        
    #     # 构建透视表: 行=Layer, 列=Proj_Type, 值=Magnitude
    #     pivot = subset.pivot_table(index="Layer", columns="Proj_Type", values="Magnitude")
        
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Magnitude'}, vmin=0)
    #     plt.title(f"Exp 3: Redundancy Map for {mod_label}", fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig(f"exp3_heatmap_modality_{mod_id}.png")
    #     plt.close()

    # ================= 实验 4: 参数分布密度 (冗余性强力证据) =================
    # 如果分布极度偏向 0，说明该模态整体微调力度很小
    print("Plotting Density Distribution...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="Magnitude", hue="Modality_Label", fill=True, common_norm=False, alpha=0.4)
    # plt.title("Exp 4: Distribution of Update Magnitudes (Peak at 0 = Redundant)", fontsize=14)
    plt.xlim(left=0)
    plt.savefig("exp4_magnitude_distribution.pdf")
    plt.close()

    print("\nAll plots saved to current directory.")

if __name__ == "__main__":
    if os.path.exists(bin_file_path):
        df_results = analyze_multimodal_lora(bin_file_path)
        plot_multimodal_analysis(df_results)
    else:
        print("File path error.")