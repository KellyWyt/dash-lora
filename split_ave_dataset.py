import os
import json

def main():
    # 配置文件路径 (请根据你的实际项目结构微调这些相对路径)
    annotations_path = 'AVE_Dataset/Annotations.txt'
    converted_label_dir = 'MokA_AudioVisualText/converted_label'
    
    train_txt_path = 'AVE_Dataset/trainSet_1.txt'
    test_txt_path = 'AVE_Dataset/testSet_1.txt'
    train_json_path = 'AVE_Dataset/train_samples_1.json'
    test_json_path = 'AVE_Dataset/test_samples_1.json'

    train_lines = []
    test_lines = []
    
    train_json_data = []
    test_json_data = []

    # 1. 读取并处理 Annotations.txt
    with open(annotations_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print("Annotations.txt 为空！")
        return

    # 提取表头
    header = lines[0].strip()
    
    # 遍历数据行
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        # 根据 '&' 分割数据
        # 格式: Category&VideoID&Quality&StartTime&EndTime
        parts = line.split('&')
        if len(parts) != 5:
            continue
            
        category, vid, quality, start_time, end_time = parts
        
        # 对应的 label txt 文件路径
        label_file_path = os.path.join(converted_label_dir, f"{vid}.txt")
        
        # 构建 JSON 格式所需的字典
        sample_dict = {
            "event": category,
            "vid": vid,
            "start_time": float(start_time),
            "end_time": float(end_time)
        }

        # 2. 判断是否存在对应的txt文件并进行划分
        if os.path.exists(label_file_path):
            train_lines.append(line)
            train_json_data.append(sample_dict)
        else:
            test_lines.append(line)
            test_json_data.append(sample_dict)

    # 3. 保存训练集和测试集的 txt 文件
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for line in train_lines:
            f.write(line + '\n')
            
    with open(test_txt_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for line in test_lines:
            f.write(line + '\n')

    # 4. 保存训练集和测试集的 json 文件
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_json_data, f, indent=4, ensure_ascii=False)
        
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_json_data, f, indent=4, ensure_ascii=False)

    print(f"处理完成！")
    print(f"训练集: 找到 {len(train_lines)} 条匹配数据。")
    print(f"测试集: 找到 {len(test_lines)} 条未匹配数据。")

if __name__ == '__main__':
    main()