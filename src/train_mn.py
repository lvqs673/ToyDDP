import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import Model
from dataset import DatasetBuilder
from comm import Communicator
from config import *
from utils import *


class Trainer:
    def __init__(
        self,
        trainset: Dataset,
        testset: Dataset,
        pred_set: Dataset,
        true_seq: list,
        communicator: Communicator,
        hidden_size: int = HIDDEN_SIZE,
        model_save_dir: str = "./model_mn",
        model_save_epoch: int = 10,
        log_path: str = "./train_mn.log",
        results_save_path: str = "./data/mn_results.json",
    ):
        self.communicator = communicator
        self.is_master = RANK == 0
        self.model = Model(hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_mn)
        # 对训练数据进行分布式采样，让每个结点训练的数据不同
        trainset = self.communicator.sample_data(trainset)
        self.train_loader = DataLoader(
            trainset, batch_size=batch_size_mn, shuffle=True)
        if self.is_master:
            self.test_loader = DataLoader(
                testset, batch_size=batch_size_mn, shuffle=False)
            self.pred_loader = DataLoader(
                pred_set, batch_size=batch_size_mn, shuffle=False)
            logging.basicConfig(
                filename=log_path,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                encoding="utf-8",
            )
        self.model_save_dir = model_save_dir
        self.model_save_epoch = model_save_epoch
        self.results_save_path = results_save_path
        self.cal_MSE = nn.MSELoss()
        self.cal_MAE = nn.L1Loss()
        self.n_epoch = n_epoch_mn

        self.true_seq = true_seq
        self.pred_seq = []
        self.train_mse_list, self.train_mae_list = [], []
        # train_moment_list记录训练开始后保存MSE和MAE时刻，用于绘制loss-time曲线
        self.train_moment_list = []
        self.test_mse_list, self.test_mae_list = [], []
        self.total_time_list = []  # 记录每个epoch训练所消耗的总时间
        self.sync_time_list = []  # 记录每个epoch同步梯度所消耗的总时间
        self.cal_time_list = []  # 记录每个epoch计算所消耗的总时间

    # 返回训练集中每个样本的MSE和MAE，以及该epoch训练总的计算耗时和通信耗时
    def train_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        self.model.train()
        train_mse, train_mae = 0.0, 0.0
        cal_time, sync_time = 0.0, 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader, 1):
            start_cal_time = time.time()
            outputs = self.model.forward(inputs)
            loss = self.cal_MSE(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            cal_time += time.time() - start_cal_time

            # 实现梯度的同步
            # print(f'({epoch},{batch_idx}) syncing grad...')
            start_sync_time = time.time()
            buckets = make_buckets(self.model)
            buckets = self.communicator.allreduce(buckets)
            change_grad(self.model, buckets)
            sync_time += time.time() - start_sync_time
            # print(f'({epoch},{batch_idx}) sync is finished')

            start_cal_time = time.time()
            self.optimizer.step()
            cal_time += time.time() - start_cal_time

            train_mse += loss.item()
            train_mae += self.cal_MAE(outputs, targets).item()

        avg_train_mse = train_mse / len(self.train_loader)
        avg_train_mae = train_mae / len(self.train_loader)
        return avg_train_mse, avg_train_mae, cal_time, sync_time

    # 返回测试集中每个样本的MSE和MAE
    def test_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        test_mse, test_mae = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader, 1):
                outputs = self.model.forward(inputs)
                test_mse += self.cal_MSE(outputs, targets).item()
                test_mae += self.cal_MAE(outputs, targets).item()
        avg_test_mse = test_mse / len(self.test_loader)
        avg_test_mae = test_mae / len(self.test_loader)
        return avg_test_mse, avg_test_mae
    
    def pred(self):
        self.model.eval()
        self.pred_seq = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.pred_loader, 1):
                outputs = self.model.forward(inputs)
                self.pred_seq.append(outputs[:,0])
        self.pred_seq = torch.cat(self.pred_seq, dim=0).tolist()

    # 存储计算的各个指标到results.json中
    def save_results(self):
        epoch_list = list(range(1, self.n_epoch + 1))
        id_seq = list(range(1, len(true_seq) + 1))
        # 单结点简记为sn，多结点简记为mn
        results = {
            "id_seq": id_seq,
            "true_seq": self.true_seq,
            "mn_pred_seq": self.pred_seq,
            "mn_epoch_list": epoch_list,
            "mn_train_mse_list": self.train_mse_list,
            "mn_train_mae_list": self.train_mae_list,
            "mn_train_moment_list": self.train_moment_list,
            "mn_test_mse_list": self.test_mse_list,
            "mn_test_mae_list": self.test_mae_list,
            "mn_total_time_list": self.total_time_list,
            "mn_sync_time_list": self.sync_time_list,
            "mn_cal_time_list": self.cal_time_list,
        }
        write_json(results, self.results_save_path)

    def out_message(self, message: str):
        logging.info(message)
        print(message)

    def save_model(self, model_name):
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        save_path = os.path.join(self.model_save_dir, model_name)
        torch.save(self.model.state_dict(), save_path)


    # 只有Master Node需要测试、存储模型和输出日志
    def train(self):
        for epoch in range(1, self.n_epoch + 1):
            train_start_time = time.time()
            train_mse, train_mae, cal_time, sync_time = self.train_epoch(epoch)
            train_cost_time = time.time() - train_start_time

            self.train_mse_list.append(train_mse)
            self.train_mae_list.append(train_mae)
            self.cal_time_list.append(cal_time)
            self.sync_time_list.append(sync_time)
            self.total_time_list.append(train_cost_time)

            if self.is_master:
                test_mse, test_mae = self.test_epoch(epoch)
                self.test_mse_list.append(test_mse)
                self.test_mae_list.append(test_mae)

                message = (
                    f"Epoch: {epoch:>2}    TrainMSE: {train_mse:>.4f}   TrainMAE: {train_mae:>.4f}"
                    + f"   TestMSE: {test_mse:>.4f}   TestMAE: {test_mae:>.4f}"
                )
                self.out_message(message)

            if self.is_master and epoch % self.model_save_epoch == 0:
                model_name = MODEL_NAME.format(epoch)
                self.save_model(model_name)

        if self.is_master:
            self.train_moment_list = np.cumsum(self.total_time_list).tolist()
            self.pred()
            self.save_results()


if __name__ == "__main__":
    builder = DatasetBuilder()
    trainset = builder.get_trainset()
    testset = builder.get_testset()
    pred_set, true_seq = builder.get_kth_dataset(PATIENT_ID)

    communicator = Communicator()
    trainer = Trainer(
        communicator=communicator,
        trainset=trainset,
        testset=testset,
        pred_set=pred_set,
        true_seq=true_seq
    )

    trainer.train()

    communicator.close()
