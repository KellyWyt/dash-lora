import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM

# ================= 配置路径 =================
# 请替换为你的实际路径
base_model_path = "/nfs1/WYT/models/Llama-2-7b-chat-hf" 
lora_weights_path = "/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"
# ===========================================

def analyze_modality_correlation():
    print(f"Loading LoRA weights from {lora_weights_path}...")
    lora_state_dict = torch.load(lora_weights_path, map_location="cpu")

    print(f"Loading Base Model (Metadata only if possible, or CPU)...")
    # 这里的 trust_remote_code=True 视情况而定
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            device_map="cpu", 
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    data = []
    
    print("Start analyzing layers...")
    for name, module in model.named_modules():
        # 只分析 Linear 层
        if not isinstance(module, nn.Linear):
            continue
            
        # 构造 LoRA 字典中的 key (根据你提供的结构，前缀是 base_model.model)
        # 原始 name 例如: model.layers.0.self_attn.q_proj
        # LoRA key 例如: base_model.model.model.layers.0.self_attn.q_proj.lora_A0.weight
        
        # 简单查找匹配
        lora_prefix = f"base_model.model.{name}"
        b0_key = f"{lora_prefix}.lora_B0.weight"
        
        if b0_key in lora_state_dict:
            # 1. 计算原始权重的重要性 (Base Norm)
            w_base = module.weight.detach().float()
            base_norm = torch.norm(w_base, p='fro').item()
            
            # 2. 获取共享的 B0
            B0 = lora_state_dict[b0_key].float() # Shape [d_out, r]
            
            # 3. 分别计算 Text(0), Video(1), Audio(2) 的更新幅度
            updates = {}
            for idx, modality in enumerate(['Text', 'Video', 'Audio']):
                a_key = f"{lora_prefix}.lora_A{idx}.weight"
                if a_key in lora_state_dict:
                    Ai = lora_state_dict[a_key].float() # Shape [r, d_in]
                    # Delta W = B @ A
                    delta_W = B0 @ Ai
                    update_norm = torch.norm(delta_W, p='fro').item()
                    updates[modality] = update_norm
                else:
                    updates[modality] = None

            # 记录数据
            if updates['Text'] is not None:
                # 提取层号用于辅助分析
                layer_id = int(name.split('.')[2]) if 'layers' in name else -1
                
                data.append({
                    'Layer Name': name,
                    'Layer ID': layer_id,
                    'Module Type': name.split('.')[-1],
                    'Base Weight Norm': base_norm,
                    'Text Update (A0)': updates['Text'],
                    'Video Update (A1)': updates['Video'],
                    'Audio Update (A2)': updates['Audio']
                })

    df = pd.DataFrame(data)
    
    # ================= 可视化分析 =================
    plt.figure(figsize=(20, 6))
    sns.set_theme(style="whitegrid")
    
    modalities = [('Text Update (A0)', 'red'), 
                  ('Video Update (A1)', 'green'), 
                  ('Audio Update (A2)', 'blue')]
    
    for i, (col_name, color) in enumerate(modalities):
        plt.subplot(1, 3, i+1)
        
        # 计算相关系数
        corr_spearman = df['Base Weight Norm'].corr(df[col_name], method='spearman')
        corr_pearson = df['Base Weight Norm'].corr(df[col_name], method='pearson')
        
        # 散点图
        sns.scatterplot(data=df, x='Base Weight Norm', y=col_name, color=color, alpha=0.6)
        # 拟合线
        sns.regplot(data=df, x='Base Weight Norm', y=col_name, scatter=False, color='black', line_kws={'linestyle':'--'})
        
        plt.title(f"{col_name.split(' ')[0]} Modality\nSpearman: {corr_spearman:.3f}, Pearson: {corr_pearson:.3f}")
        plt.xlabel("Base Weight Importance (Norm)")
        plt.ylabel("LoRA Update Magnitude")
    
    plt.tight_layout()
    plt.savefig("multimodal_lora_analysis.png")
    print("\nAnalysis complete. Plot saved to 'multimodal_lora_analysis.png'.")
    
    # ================= 打印具体数值结论 =================
    print("\n====== Correlation Summary ======")
    print(df[['Base Weight Norm', 'Text Update (A0)', 'Video Update (A1)', 'Audio Update (A2)']].corr(method='spearman')['Base Weight Norm'])
    
    return df

# 执行分析
if __name__ == "__main__":
    df_result = analyze_modality_correlation()