#!/bin/bash

# 实验脚本：测试不同的one_hop_weight参数
# 运行5组实验，权重分别为：1.5, 2.0, 2.5, 3.0, 3.5

echo "=========================================="
echo "开始运行one_hop_weight参数实验" 串行
echo "实验组数：5组"
echo "参数值：1.5, 2.0, 2.5, 3.0, 3.5"
echo "=========================================="
echo ""

# 定义one_hop_weight参数列表
weights=(1.5 2.0 2.5 3.0 3.5)

# 记录总开始时间
total_start=$(date +%s)

# 循环运行实验
for i in "${!weights[@]}"
do
    weight=${weights[$i]}
    exp_num=$((i+1))
    
    echo "=========================================="
    echo "实验 $exp_num/5: one_hop_weight = $weight"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    # 记录单个实验开始时间
    exp_start=$(date +%s)
    
    # 运行Python脚本
    python Main.py --one_hop_weight $weight
    
    # 记录单个实验结束时间
    exp_end=$(date +%s)
    exp_elapsed=$((exp_end - exp_start))
    exp_minutes=$((exp_elapsed / 60))
    exp_seconds=$((exp_elapsed % 60))
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "✅ 实验 $exp_num 完成 (one_hop_weight=$weight)"
        echo "⏱️  耗时: ${exp_minutes}分${exp_seconds}秒"
    else
        echo "❌ 实验 $exp_num 失败 (one_hop_weight=$weight)"
        echo "⏱️  耗时: ${exp_minutes}分${exp_seconds}秒"
    fi
    
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

# 记录总结束时间
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))
total_hours=$((total_elapsed / 3600))
total_minutes=$(((total_elapsed % 3600) / 60))
total_seconds=$((total_elapsed % 60))

echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
echo "总实验数: 5组"
echo "总耗时: ${total_hours}小时 ${total_minutes}分钟 ${total_seconds}秒"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
