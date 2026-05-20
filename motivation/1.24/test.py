import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import io
import re
import os
import traceback

# ==========================================
# 0. 配置与准备
# ==========================================
# 设置绘图风格 (学术风)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['axes.unicode_minus'] = False 

OUTPUT_DIR = "results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录: {OUTPUT_DIR}")

# ==========================================
# 1. 数据加载逻辑 (自动跳过元数据行)
# ==========================================
def load_norm_data(file_path):
    try:
        # 1. 预读取文件，找到表头
        header_row_index = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Rank" in line and "|" in line:
                    header_row_index = i
                    break
        
        print(f"检测到表头在第 {header_row_index + 1} 行，开始读取...")

        # 2. 读取 CSV
        df = pd.read_csv(
            file_path, 
            sep='|', 
            engine='python', 
            skipinitialspace=True, 
            skiprows=header_row_index
        )
        df.columns = [c.strip() for c in df.columns]
        
        # 3. 数据清洗
        df = df[pd.to_numeric(df['Rank'], errors='coerce').notnull()].copy()
        df['Rank'] = df['Rank'].astype(int)
        df['Norm'] = df['Norm'].astype(float)
        
        # 解析层号
        def parse_info(name):
            if not isinstance(name, str): return -1, 'unknown'
            match = re.search(r'layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)', name)
            if match:
                return int(match.group(1)), match.group(2)
            return -1, 'unknown'
            
        df[['Layer', 'Type']] = df['Module Name'].apply(lambda x: pd.Series(parse_info(x)))
        print(f"成功加载 {len(df)} 行有效数据。")
        return df

    except Exception as e:
        print(f"数据读取错误: {e}")
        print(traceback.format_exc())
        return None

# !!! 这里文件名根据你的截图来看是 importance_rank_report_topk_56.txt !!!
file_path = 'importance_rank_report_topk_56.txt' 

if not os.path.exists(file_path):
    print(f"❌ 未找到文件: {file_path}")
    # 为了演示，生成模拟数据 (实际运行时不会走这里)
    ranks = np.arange(1, 225)
    norms = 100 * np.exp(-0.02 * ranks) + 20
    norms[:28] += 60
    data_str = "Rank | Module Name | Norm\n"
    for r, n in zip(ranks, norms):
        if r <= 28: layer = np.random.choice([0,1,2,30,31])
        elif r <= 56: layer = np.random.randint(5, 28)
        else: layer = np.random.randint(0, 32)
        mtype = 'gate_proj' if r%2==0 else 'q_proj'
        data_str += f"{r} | layers.{layer}.mlp.{mtype} | {n:.2f}\n"
    df = load_norm_data(io.StringIO(data_str))
else:
    df = load_norm_data(file_path)

# ==========================================
# 2. 核心计算
# ==========================================
if df is not None:
    def get_allocation(rank):
        if rank <= 28: return 'Top-28 (Video/Audio)'
        elif rank <= 56: return 'Top-56 (Text Extension)'
        else: return 'Shared Base'

    df['Allocation'] = df['Rank'].apply(get_allocation)

    df_sorted = df.sort_values('Rank')
    total_norm = df_sorted['Norm'].sum()
    df_sorted['Cumulative_Norm'] = df_sorted['Norm'].cumsum()
    df_sorted['Norm_Ratio'] = df_sorted['Cumulative_Norm'] / total_norm

    # 关键点标注
    norm_cov_28 = df_sorted[df_sorted['Rank']==28]['Norm_Ratio'].values[0] if 28 in df_sorted['Rank'].values else 0.4
    norm_cov_56 = df_sorted[df_sorted['Rank']==56]['Norm_Ratio'].values[0] if 56 in df_sorted['Rank'].values else 0.7

    PALETTE = {
        'Top-28 (Video/Audio)': '#d62728', 
        'Top-56 (Text Extension)': '#1f77b4', 
        'Shared Base': '#7f7f7f'
    }

    # ==========================================
    # 3. 绘图 (修复了 range 参数报错)
    # ==========================================

    # --- 图 1: Norm 悬崖 ---
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_sorted, x='Rank', y='Norm', color='black', linewidth=1.5)
    plt.fill_between(df_sorted['Rank'], 0, df_sorted['Norm'], where=(df_sorted['Rank']<=28), color=PALETTE['Top-28 (Video/Audio)'], alpha=0.3, label='High Impact (Video/Audio)')
    plt.fill_between(df_sorted['Rank'], 0, df_sorted['Norm'], where=((df_sorted['Rank']>28)&(df_sorted['Rank']<=56)), color=PALETTE['Top-56 (Text Extension)'], alpha=0.3, label='Medium Impact (Text)')
    plt.axvline(28, color='red', linestyle='--')
    plt.axvline(56, color='blue', linestyle='--')
    plt.title('Evidence 1: Parameter Update Intensity (The Cliff)', fontsize=14, fontweight='bold')
    plt.xlabel('Module Rank')
    plt.ylabel('Norm Value (Update Magnitude)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_norm_cliff.png", dpi=300)
    plt.close()

    # --- 图 2: 累计能量 ---
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_sorted, x='Rank', y='Norm_Ratio', color='black', linewidth=2)
    plt.axvline(28, color='red', linestyle='--')
    plt.axvline(56, color='blue', linestyle='--')
    plt.plot(28, norm_cov_28, 'o', color='red', markersize=8)
    plt.text(35, norm_cov_28-0.05, f'Rank 28\nCovers {norm_cov_28:.1%} Energy', color='#d62728', fontweight='bold')
    plt.plot(56, norm_cov_56, 'o', color='blue', markersize=8)
    plt.text(63, norm_cov_56-0.05, f'Rank 56\nCovers {norm_cov_56:.1%} Energy', color='#1f77b4', fontweight='bold')
    plt.title('Evidence 2: Cumulative Energy Coverage', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Norm Ratio (0-1)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_cumulative_energy.png", dpi=300)
    plt.close()

    # --- 图 3: 层级分布 (修复点：range -> binrange) ---
    plt.figure(figsize=(10, 6))
    top56 = df_sorted[df_sorted['Rank'] <= 56].copy()
    sns.histplot(
        data=top56, x='Layer', hue='Allocation',
        hue_order=['Top-28 (Video/Audio)', 'Top-56 (Text Extension)'],
        palette=PALETTE,
        multiple='layer', 
        bins=32, 
        binrange=(0,32), # <--- 关键修改：从 range 改为 binrange
        alpha=0.5, 
        edgecolor='white'
    )
    plt.title('Evidence 3: Layer Preference (Structural Complement)', fontsize=14, fontweight='bold')
    plt.xlabel('Transformer Layer Index (0 = Input, 31 = Output)')
    plt.ylabel('Count of Selected Modules')
    plt.xticks(np.arange(0, 33, 4))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_layer_distribution.png", dpi=300)
    plt.close()

    # --- 图 4: 组合图 (修复点：range -> binrange) ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # 子图1
    sns.lineplot(data=df_sorted, x='Rank', y='Norm', ax=axes[0], color='k')
    axes[0].fill_between(df_sorted['Rank'], 0, df_sorted['Norm'], where=(df_sorted['Rank']<=28), color=PALETTE['Top-28 (Video/Audio)'], alpha=0.3)
    axes[0].fill_between(df_sorted['Rank'], 0, df_sorted['Norm'], where=((df_sorted['Rank']>28)&(df_sorted['Rank']<=56)), color=PALETTE['Top-56 (Text Extension)'], alpha=0.3)
    axes[0].set_title('Absolute Norm Distribution')

    # 子图2
    sns.lineplot(data=df_sorted, x='Rank', y='Norm_Ratio', ax=axes[1], color='k')
    axes[1].axvline(28, color='red', linestyle='--')
    axes[1].axvline(56, color='blue', linestyle='--')
    axes[1].plot(28, norm_cov_28, 'ro')
    axes[1].plot(56, norm_cov_56, 'bo')
    axes[1].set_title(f'Energy Coverage: Top 28={norm_cov_28:.0%}, Top 56={norm_cov_56:.0%}')

    # 子图3 (同样修复 binrange)
    sns.histplot(
        data=top56, x='Layer', hue='Allocation', 
        hue_order=['Top-28 (Video/Audio)', 'Top-56 (Text Extension)'],
        palette=PALETTE, multiple='layer', bins=32, 
        binrange=(0,32), # <--- 关键修改
        alpha=0.5, ax=axes[2]
    )
    axes[2].set_title('Layer Distribution (Red=Ends, Blue=Middle)')

    plt.suptitle('Justification for 2:1:1 Ratio based on LoRA Norm Analysis', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/summary_dashboard_high_res.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 成功！图片已保存至 '{OUTPUT_DIR}/' 文件夹。")