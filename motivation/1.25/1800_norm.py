import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Load weights
bin_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"
state_dict = torch.load(bin_path, map_location="cpu")

# Define parameters
num_layers = 32
modalities = ['Text', 'Video', 'Audio']

# modalities = ['Text (A0)', 'Video (A1)', 'Audio (A2)']
modality_keys = ['lora_A0', 'lora_A1', 'lora_A2']
shared_b_key = 'lora_B0'

# Use nested lists to store scores for confidence interval calculation
# Structure: {modality: [layers, module_scores]}
raw_scores = {m: [[] for _ in range(num_layers)] for m in modalities}

for i in range(num_layers):
    layer_pattern = f"layers.{i}."
    layer_keys = [k for k in state_dict.keys() if layer_pattern in k]
    
    module_prefixes = set()
    for k in layer_keys:
        if shared_b_key in k:
            prefix = k.split(shared_b_key)[0]
            module_prefixes.add(prefix)
            
    for prefix in module_prefixes:
        key_B = prefix + f"{shared_b_key}.weight"
        if key_B not in state_dict: continue
        
        B = state_dict[key_B].float()
        
        for m_idx, m_name in enumerate(modalities):
            key_A = prefix + f"{modality_keys[m_idx]}.weight"
            
            if key_A in state_dict:
                A = state_dict[key_A].float()
                delta_W = torch.matmul(B, A)
                norm = torch.norm(delta_W).item()
                raw_scores[m_name][i].append(norm)

# Process stats for visualization
final_means = {m: np.zeros(num_layers) for m in modalities}
final_stds = {m: np.zeros(num_layers) for m in modalities}

for m in modalities:
    for i in range(num_layers):
        if raw_scores[m][i]:
            final_means[m][i] = np.mean(raw_scores[m][i])
            final_stds[m][i] = np.std(raw_scores[m][i])

# --- Academic Visualization ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

plt.figure(figsize=(10, 6))
colors = sns.color_palette("muted", n_colors=3)
markers = ['o', 's', 'D']

x = np.arange(num_layers)

for idx, m_name in enumerate(modalities):
    mean_val = final_means[m_name]
    std_val = final_stds[m_name]
    
    # Plot mean line
    plt.plot(x, mean_val, label=m_name, color=colors[idx], 
             marker=markers[idx], markersize=5, linewidth=2, alpha=0.9)
    
    # Plot confidence/variance area
    plt.fill_between(x, mean_val - std_val, mean_val + std_val, 
                     color=colors[idx], alpha=0.2)

# Axis labels
plt.xlabel("Llama Layer Index")
plt.ylabel("Weight Norm")

# Highlight specific regions
plt.axvspan(0, 4, color='gray', alpha=0.05)
plt.axvspan(28, 31, color='gray', alpha=0.05)

plt.xlim(-0.5, 31.5)
plt.legend(frameon=True, loc='upper right', fontsize='small')
plt.tight_layout()

plt.savefig("modality_importance_analysis.pdf", dpi=600, bbox_inches='tight')
plt.show()