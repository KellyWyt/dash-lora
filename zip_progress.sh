TARGET_DIR="./music-avqa"
# 生成的压缩包名字
ZIP_NAME="./music-avqa.zip"
# =========================================

# 1. 检查目标是否存在
if [ ! -d "$TARGET_DIR" ]; then
	    echo "错误: 找不到文件夹 $TARGET_DIR"
	        exit 1
fi

echo "正在扫描文件总数，请稍候..."
# 统计文件总数 (find 命令)
TOTAL_FILES=$(find "$TARGET_DIR" | wc -l)

echo "检测到共有 $TOTAL_FILES 个文件/文件夹，开始压缩..."
echo "目标: $ZIP_NAME"
echo ""

# 2. 核心逻辑
# zip -r 正常运行，把输出管道传给 awk
# awk 负责计算：当前行数 / 总行数 = 百分比
# \r 是回车符，让进度条在同一行刷新，不会刷屏
zip -r "$ZIP_NAME" "$TARGET_DIR" | \
	awk -v total="$TOTAL_FILES" '{
    percent = (NR / total) * 100
        printf "\r[%-50s] %6.2f%% (%d/%d)", border, percent, NR, total
	    # 简单的进度条效果 (每2%加一个#)
	        if(int(percent/2) > length(border)) {
			        for(i=length(border); i<int(percent/2); i++) border=border "#"
					    }
			    }'

		    echo -e "\n\n压缩完成！文件已保存为: $ZIP_NAME"
