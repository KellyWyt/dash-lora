import torch
import os

# ==========================================
# 1. 请在这里填入你的两个 bin 文件的路径
# ==========================================
bin_file_path_1 = r"/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/finetune_weights.bin"   # 建议使用 r"" 防止路径转义
bin_file_path_2 = r"/nfs1/WYT/MokA-copy/results/finetune/llama_music/adapter_model.bin"

# ==========================================
# 功能实现代码
# ==========================================

def save_bin_structure(file_path):
    # 1. 检查输入文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到输入文件 -> {file_path}")
        return

    # 2. 构造输出文件名
    # 获取文件名 (如 'pytorch_model.bin')
    base_name = os.path.basename(file_path)
    # 构造新名字 (如 'pytorch_model.bin_save.txt')
    output_filename = f"{base_name}_save.txt"
    # 获取当前运行目录，确保文件保存在当前脚本目录下，或者你可以指定绝对路径
    output_path = os.path.join(os.getcwd(), output_filename)

    print(f"🔄 正在处理: {base_name} ...")

    try:
        # 3. 加载模型 (CPU 模式)
        state_dict = torch.load(file_path, map_location='cpu', weights_only=True)
        
        # 4. 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            # 写入头部信息
            f.write(f"原文件路径: {file_path}\n")
            f.write(f"参数层总数: {len(state_dict)}\n")
            f.write("=" * 100 + "\n")
            # 写入表头
            header = f"{'层名称 (Layer Name)':<60} | {'尺寸 (Shape)':<25} | {'类型 (Dtype)'}"
            f.write(header + "\n")
            f.write("-" * 100 + "\n")

            # 遍历并写入所有层
            for key, value in state_dict.items():
                if hasattr(value, 'shape'):
                    shape_str = str(list(value.shape))
                    dtype_str = str(value.dtype).replace('torch.', '')
                    line = f"{key:<60} | {shape_str:<25} | {dtype_str}\n"
                    f.write(line)
                else:
                    # 处理非 Tensor 数据
                    line = f"{key:<60} | {'Non-Tensor':<25} | {type(value)}\n"
                    f.write(line)
        
        print(f"✅ 保存成功! 结果已写入 -> {output_filename}")

    except Exception as e:
        print(f"❌ 处理文件 {base_name} 时发生错误: {e}")

# ==========================================
# 执行
# ==========================================
print("开始任务...\n")
save_bin_structure(bin_file_path_1)
save_bin_structure(bin_file_path_2)
print("\n任务完成。")