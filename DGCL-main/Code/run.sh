#!/bin/bash

# 通用超参实验脚本
# 当前控制参数：--lr
# 以后想换其它参数（比如 --batch、--reg），只要改 PARAM_NAME 和 PARAM_VALUES 即可

# 要控制的参数名（与 Params.py 中保持一致）
PARAM_NAME="latdim"

# 要实验的一组取值（示例：学习率）
PARAM_VALUES=(16 32 64 256)

# 其它你想固定的参数（可选）
DATASET="DGIdb"      # 或 DrugBank
GPU_ID=0             # 对应 Params.py 里的 --gpu
EXTRA_ARGS="--data ${DATASET} --gpu ${GPU_ID}"

echo "=========================================="
echo "开始运行超参数实验：--${PARAM_NAME} in (${PARAM_VALUES[*]})"
echo "数据集: ${DATASET}, GPU: ${GPU_ID}"
echo "=========================================="
echo ""

total_exps=${#PARAM_VALUES[@]}
echo "实验组数：${total_exps} 组"

total_start=$(date +%s)

for i in "${!PARAM_VALUES[@]}"; do
    value=${PARAM_VALUES[$i]}
    exp_num=$((i + 1))

    echo "------------------------------------------"
    echo "实验 ${exp_num}/${total_exps}"
    echo "--${PARAM_NAME} = ${value}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    exp_start=$(date +%s)

    # 调用主训练脚本：只改 --${PARAM_NAME}，其它超参都走 Params.py 默认逻辑
    python Main.py --${PARAM_NAME} "${value}" ${EXTRA_ARGS}

    status=$?
    exp_end=$(date +%s)
    exp_elapsed=$((exp_end - exp_start))
    exp_min=$((exp_elapsed / 60))
    exp_sec=$((exp_elapsed % 60))

    if [ ${status} -eq 0 ]; then
        echo "✅ 实验 ${exp_num} 完成 (--${PARAM_NAME}=${value})"
    else
        echo "❌ 实验 ${exp_num} 失败 (--${PARAM_NAME}=${value})"
    fi
    echo "⏱️  耗时: ${exp_min} 分 ${exp_sec} 秒"
    echo ""
done

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))
total_hours=$((total_elapsed / 3600))
total_minutes=$(((total_elapsed % 3600) / 60))
total_seconds=$((total_elapsed % 60))

echo "=========================================="
echo "🎉 所有实验完成！"
echo "总实验数: ${total_exps} 组"
echo "总耗时: ${total_hours} 小时 ${total_minutes} 分钟 ${total_seconds} 秒"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="