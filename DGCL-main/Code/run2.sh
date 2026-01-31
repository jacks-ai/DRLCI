#!/bin/bash

# 实验脚本：测试one_hop_weight和two_hop_weight的组合参数
# one_hop_weight: 2, 2.5, 3
# two_hop_weight: 1.5, 2, 2.5
# 总共9组实验

echo "=========================================="
echo "开始运行one_hop_weight和two_hop_weight组合参数实验"
echo "实验组数：9组"
echo "one_hop_weight参数值：2.0, 2.5, 3.0"
echo "two_hop_weight参数值：1.5, 2.0, 2.5"
echo "=========================================="
echo ""

# 定义参数列表
one_hop_weights=(2.0 2.5 3.0)
two_hop_weights=(1.5 2.0 2.5)

# 记录总开始时间
total_start=$(date +%s)

# 实验计数器
exp_num=0
total_exps=$((${#one_hop_weights[@]} * ${#two_hop_weights[@]}))

# 双重循环运行实验
for one_hop in "${one_hop_weights[@]}"
do
    for two_hop in "${two_hop_weights[@]}"
    do
        exp_num=$((exp_num + 1))
        
        echo "=========================================="
        echo "实验 $exp_num/$total_exps"
        echo "one_hop_weight = $one_hop"
        echo "two_hop_weight = $two_hop"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        
        # 记录单个实验开始时间
        exp_start=$(date +%s)
        
        # 运行Python脚本
        python Main.py --one_hop_weight $one_hop --two_hop_weight $two_hop
        
        # 记录单个实验结束时间
        exp_end=$(date +%s)
        exp_elapsed=$((exp_end - exp_start))
        exp_minutes=$((exp_elapsed / 60))
        exp_seconds=$((exp_elapsed % 60))
        
        # 检查退出状态
        if [ $? -eq 0 ]; then
            echo "✅ 实验 $exp_num 完成 (one_hop=$one_hop, two_hop=$two_hop)"
            echo "⏱️  耗时: ${exp_minutes}分${exp_seconds}秒"
        else
            echo "❌ 实验 $exp_num 失败 (one_hop=$one_hop, two_hop=$two_hop)"
            echo "⏱️  耗时: ${exp_minutes}分${exp_seconds}秒"
        fi
        
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    done
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
echo "总实验数: $total_exps 组"
echo "总耗时: ${total_hours}小时 ${total_minutes}分钟 ${total_seconds}秒"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
