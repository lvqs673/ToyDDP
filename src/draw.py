import os
import matplotlib.pyplot as plt
from utils import *

"""
该文件是为了绘制可视化的图像
从./data/mn_results.json和./data/sn_results.json中取数据绘制


六种图
1. 训练集Loss-Epoch曲线
训练集上单结点和多结点的MAE随Epoch变化的曲线
2. 测试集Loss-Epoch曲线
测试集上单结点和多结点的MAE随Epoch变化的曲线
3. 训练集Loss-Time曲线
训练集上单结点和多结点的MAE随Time变化的曲线
4. 预测曲线
第100个患者的真实曲线、单结点模型预测曲线、多结点模型预测曲线
5. 单结点训练每个Epoch训练时长、多结点Master Node每个Epoch计算时长+同步时长

"""




# 两个曲线在同一张图
def draw2(
    x: list, y1: list, y2: list, x2: list=None,
    x_label: str = None, y_label: str = None,
    y1_name: str = None, y2_name: str = None,
    title: str = None, save_name: str = None, marker=True,
):
    plt.figure(figsize=(10, 6))
    x2 = x if x2 is None else x2
    if marker:
        plt.plot(x, y1, label=y1_name, color="darkorange", linewidth=2, marker="o", markersize=4)
        plt.plot(x2, y2, label=y2_name, color="dodgerblue", linewidth=2, marker="^", markersize=4)
    else:
        plt.plot(x, y1, label=y1_name, color="darkorange", linewidth=2)
        plt.plot(x2, y2, label=y2_name, color="dodgerblue", linewidth=2)
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    if y1_name or y2_name:
        plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_name:
        save_path = os.path.join(IMAGE_DIR, save_name)
        plt.savefig(save_path)



sn_results = read_json("./data/sn_results.json")
mn_results = read_json("./data/mn_results.json")

# 训练集Loss-Epoch曲线
draw2(
    x=mn_results["mn_epoch_list"],
    y1=sn_results["sn_train_mae_list"],
    y2=mn_results["mn_train_mae_list"],
    y1_name="sn_train_mae",
    y2_name="mn_train_mae",
    x_label="Epoch",
    y_label="MAELoss",
    title="TrainLoss-Epoch",
    save_name="TrainLoss-Epoch.jpg",
)
#  测试集Loss-Epoch曲线
draw2(
    x=mn_results["mn_epoch_list"],
    y1=sn_results["sn_test_mae_list"],
    y2=mn_results["mn_test_mae_list"],
    y1_name="sn_test_mae",
    y2_name="mn_test_mae",
    x_label="Epoch",
    y_label="MAELoss",
    title="TestLoss-Epoch",
    save_name="TestLoss-Epoch.jpg",
)
# 训练集Loss-Time曲线
draw2(
    x=sn_results["sn_train_moment_list"],
    x2=mn_results["mn_train_moment_list"],
    y1=sn_results["sn_train_mae_list"],
    y2=mn_results["mn_train_mae_list"],
    y1_name="sn_train_mae",
    y2_name="mn_train_mae",
    x_label="Time (second)",
    y_label="MAELoss",
    title="TrainLoss-Time",
    save_name="TrainLoss-Time.jpg",
)


# 三个曲线在同一张图
def draw3(
    x: list, y1: list, y2: list, y3: list, 
    x_label: str = None, y_label: str = None,
    y1_name: str = None, y2_name: str = None, y3_name: str = None,
    title: str = None, save_name: str = None, marker=True,
):
    plt.figure(figsize=(10, 6))
    if marker:
        plt.plot(x, y1, label=y1_name, color="darkorange", linewidth=2, marker="o", markersize=4)
        plt.plot(x, y2, label=y2_name, color="dodgerblue", linewidth=2, marker="^", markersize=4)
        plt.plot(x, y3, label=y3_name, color="mediumseagreen", linewidth=2, linestyle='-.', marker="s", markersize=5)
    else:
        plt.plot(x, y1, label=y1_name, color="darkorange", linewidth=2)
        plt.plot(x, y2, label=y2_name, color="dodgerblue", linewidth=2)
        plt.plot(x, y3, label=y3_name, color="mediumseagreen", linewidth=2)
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    if y1_name or y2_name or y3_name:
        plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_name:
        save_path = os.path.join(IMAGE_DIR, save_name)
        plt.savefig(save_path)

# 第100个患者的血糖预测曲线
draw3(
    x=mn_results["id_seq"][:NUM_PREDICTION_POINTS],
    y1=mn_results["true_seq"][:NUM_PREDICTION_POINTS],
    y2=sn_results["sn_pred_seq"][:NUM_PREDICTION_POINTS],
    y3=mn_results["mn_pred_seq"][:NUM_PREDICTION_POINTS],
    x_label="CheckPointId",
    y_label="Glucose (mmol/L)",
    y1_name="true_seq",
    y2_name="sn_pred_seq",
    y3_name="mn_pred_seq",
    title="GlucosePredictionCurve",
    save_name="GlucosePredictionCurve.jpg",
    marker=False,
)


def draw_time(sn_total_time: float, mn_cal_time:float, mn_sync_time: float, save_name: str=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_width = 0.2
    positions = [0.2, 0.8]
    ax.set_xlim(-0.2, 1.2)
    sn_bars = ax.bar(positions[0], sn_total_time, bar_width, label='Computing', color='green')
    mn_cal_bars = ax.bar(positions[1], mn_cal_time, bar_width, color='green')
    mn_sync_bars = ax.bar(positions[1], mn_sync_time, bar_width, bottom=mn_cal_time, label='Syncing', color='lightgreen')
    ax.set_ylabel('Time (Seconds)')
    ax.set_title('Single Node vs Multi Node in Training Times')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Single Node', 'Multi Node'])
    ax.legend()
    if save_name:
        save_path = os.path.join(IMAGE_DIR, save_name)
        plt.savefig(save_path)


# sn_total_time为单结点训练时每个epoch的总时长
sn_total_time =mean(sn_results["sn_total_time_list"])

# mn_cal_time为多结点训练时每个epoch的计算花费时长
# mn_sync_time为多结点训练时每个epoch的梯度同步花费时长
mn_cal_time = mean(mn_results["mn_cal_time_list"])
mn_sync_time = mean(mn_results["mn_sync_time_list"])

draw_time(
    sn_total_time=sn_total_time,
    mn_cal_time=mn_sync_time,
    mn_sync_time=mn_sync_time,
    save_name="TrainingTimeComparison.jpg"
)