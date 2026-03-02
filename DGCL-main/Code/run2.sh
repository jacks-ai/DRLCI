#!/bin/bash

# 实验脚本：测试one_hop_weight和two_hop_weight的配对参数
# 实验1: one_hop_weight=2.0, two_hop_weight=1.5
# 实验2: one_hop_weight=2.5, two_hop_weight=2.0
# 实验3: one_hop_weight=3.0, two_hop_weight=2.5
# 总共3组实验

echo "=========================================="
echo "开始运行one_hop_weight和two_hop_weight配对参数实验 one_hop_weights=(3.0 4.0 5.0) two_hop_weights=(2.0 3.0 4.0)"

# 定义配对参数列表
#one_hop_weights=(2.5 4.5 3.5)
#two_hop_weights=(1.5 3.5 2.5)

one_hop_weights=(2.5 3.0)
two_hop_weights=(1.5 2.0)


# 总实验数（动态计算）
total_exps=${#one_hop_weights[@]}

echo "实验组数：${total_exps}组"
echo "=========================================="
echo ""

# 记录总开始时间
total_start=$(date +%s)

# 循环运行配对实验
for i in "${!one_hop_weights[@]}"
do
    one_hop=${one_hop_weights[$i]}
    two_hop=${two_hop_weights[$i]}
    exp_num=$((i + 1))
    
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


# 记录总结束时间
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))
total_hours=$((total_elapsed / 3600))
total_minutes=$(((total_elapsed % 3600) / 60))
total_seconds=$((total_elapsed % 60))

echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
echo "总实验数: ${total_exps} 组"
echo "总耗时: ${total_hours}小时 ${total_minutes}分钟 ${total_seconds}秒"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
