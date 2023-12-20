import os
import matplotlib.pyplot as plt
from utils import *

"""
该文件是为了绘制可视化的图像
从./data/results.json中取数据绘制
其中的数据：
id_seq
true_seq
sn_pred_seq
epoch_list
sn_train_mse_list
sn_train_mae_list
sn_test_mse_list
sn_test_mae_list
sn_total_time_list


将单结点和多结点train_mse、train_mae、test_mse、test_mae放到四个图中，
    绘制TrainMSE-Epoch、TrainMAE-Epoch、TestMSE-Epoch、TestMAE-Epoch四个曲线
将单结点平均每个epoch总耗时和多结点平均每个epoch总耗时放到一个图中比较，多结点的计算耗时在下、通信耗时在上（TimeCosted）
将单结点模型和多结点模型对100号患者的预测曲线和真实曲线绘制在一张图中（PredictionCurve）
"""


# 一个曲线
def draw1(
    x: list,
    y: list,
    x_label: str = None,
    y_label: str = None,
    y_name: str = None,
    title: str = None,
    save_path: str = None,
    marker=True,
):
    plt.figure(figsize=(10, 6))  # 设置图表大小

    if marker:
        plt.plot(
            x,
            y,
            label=y_name,
            color="dodgerblue",
            linewidth=2,
            marker="o",
            markersize=4,
        )
    else:
        plt.plot(x, y, label=y_name, color="dodgerblue", linewidth=2)

    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.5)

    if y_name:
        plt.legend(loc="best", fontsize=10)

    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# 两个曲线在同一张图
def draw2(
    x: list,
    y1: list,
    y2: list,
    x_label: str = None,
    y_label: str = None,
    y1_name: str = None,
    y2_name: str = None,
    title: str = None,
    save_path: str = None,
    marker=True,
):
    plt.figure(figsize=(10, 6))

    if marker:
        plt.plot(
            x,
            y1,
            label=y1_name,
            color="darkorange",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.plot(
            x,
            y2,
            label=y2_name,
            color="dodgerblue",
            linewidth=2,
            marker="^",
            markersize=4,
        )
    else:
        plt.plot(x, y1, label=y1_name, color="darkorange", linewidth=2)
        plt.plot(x, y2, label=y2_name, color="dodgerblue", linewidth=2)

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

    if save_path:
        plt.savefig(save_path)

    plt.show()


results = read_json("./data/results.json")

draw1(
    x=results["epoch_list"],
    y=results["sn_train_mse_list"],
    x_label="Epoch/代",
    y_label="Loss",
    title="TrainMSE-Epoch",
    save_path="a.jpg",
)
draw2(
    x=results["epoch_list"],
    y1=results["sn_train_mae_list"],
    y2=results["sn_test_mae_list"],
    y1_name="train_mae",
    y2_name="test_mae",
    x_label="Epoch",
    y_label="Loss",
    title="MAE-Epoch",
    save_path="b.jpg",
)
n = 100
draw2(
    x=results["id_seq"][:n],
    y1=results["true_seq"][:n],
    y2=results["sn_pred_seq"][:n],
    y1_name="true_seq",
    y2_name="sn_pred_seq",
    x_label="Id",
    y_label="Glucose",
    title="GlucosePrediction",
    save_path="c.jpg",
    marker=False,
)
