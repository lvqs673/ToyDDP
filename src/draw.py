


"""
该文件是为了绘制可视化的图像
从./data/results.json中取数据绘制
其中的数据：
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


