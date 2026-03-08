import matplotlib
# 如果 PyCharm 报错，请取消下面这行的注释
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 ---
labels = ['DRLCI', 'MNCL', 'DGCL', 'CosMIG']
acc_values = [0.637, 0.626, 0.615, 0.575]

# --- 2. 自动计算 Improvement (左边相比右边提高了多少) ---
improvements = []
for i in range(len(acc_values)):
    if i == len(acc_values) - 1:
        improvements.append("(0%)")
    else:
        # 计算当前项相对于其右侧相邻项的提升（或者统一相对于最右侧 CosMIG）
        # 这里统一相对于最右侧基准 CosMIG 计算总提升，符合原图视觉逻辑
        base_val = acc_values[-1]
        diff = (acc_values[i] - base_val) / base_val * 100
        improvements.append(f"↑{diff:.2f}%")

# --- 3. 绘图配置 ---
# 颜色方案：深紫 -> 中紫 -> 浅紫 -> 灰色
colors = ['#5A248E', '#8E7CC3', '#C2B7E1', '#E8E8E8']
line_color = '#F9A825'

# 缩小宽度 (figsize的宽度调小可以压缩柱间距)
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

# width=0.5 让柱子变瘦，ax.set_xlim 手动控制左右留白防止间距过大
bars = ax.bar(labels, acc_values, color=colors, edgecolor='black', linewidth=0.6, width=0.5)

# --- 4. 绘制折线与星号 ---
ax.plot(labels, acc_values, marker='*', markersize=14, markerfacecolor=line_color,
        markeredgecolor='white', color=line_color, linestyle='--', linewidth=2.5, label='Improvement (%)')

# --- 5. 添加浮动文本框 ---
for i, (bar, text) in enumerate(zip(bars, improvements)):
    height = bar.get_height()

    # 样式判断
    if "Baseline" in text:
        edge_c, face_c, text_c = '#F57C00', '#FFE0B2', '#E65100'
    else:
        edge_c, face_c, text_c = '#2E7D32', '#C8E6C9', '#2E7D32'

    ax.annotate(text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=text_c,
                bbox=dict(boxstyle='round,pad=0.3', edgecolor=edge_c, facecolor=face_c, alpha=0.9))

# --- 6. 坐标轴与细节微调 ---
ax.set_title('LINCS L1000 Performance', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('ACC', fontsize=12, fontweight='bold')

# 按照要求固定纵坐标范围
ax.set_ylim(0.56, 0.66)
# 适当收紧横轴范围，使柱子靠得更近
ax.set_xlim(-0.5, len(labels) - 0.5)

ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
ax.set_axisbelow(True)

plt.xticks(fontsize=11, fontweight='bold')

# 图例（带阴影）
ax.legend(loc='upper right', shadow=True, prop={'weight': 'bold'})

plt.tight_layout()
plt.savefig('custom_style_plot.png', dpi=300)
plt.show()