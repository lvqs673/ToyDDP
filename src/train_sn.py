import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model import Model
from dataset import DatasetBuilder
from config import *
from utils import *


class Trainer:
    def __init__(
        self,
        trainset: Dataset,
        testset: Dataset,
        hidden_size: int = HIDDEN_SIZE,
        model_save_dir: str = "./model_sn",
        model_save_epoch: int = 10,
        log_path: str = "./train_sn.log",
        results_save_path: str = "./data/results.json",
    ):
        self.model = Model(hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_sn)
        self.train_loader = DataLoader(
            trainset, batch_size=batch_size_sn, shuffle=True)
        self.test_loader = DataLoader(
            testset, batch_size=batch_size_sn, shuffle=False)
        self.model_save_dir = model_save_dir
        self.model_save_epoch = model_save_epoch
        self.results_save_path = results_save_path
        self.cal_MSE = nn.MSELoss()
        self.cal_MAE = nn.L1Loss()
        self.n_epoch = n_epoch_sn

        logging.basicConfig(
            filename=log_path,
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            encoding="utf-8",
        )

        self.train_mse_list, self.train_mae_list = [], []
        self.test_mse_list, self.test_mae_list = [], []
        self.total_time_list = []  # 记录每个epoch训练所消耗的总时间

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        train_mse, train_mae = 0.0, 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader, 1):
            outputs = self.model.forward(inputs)
            loss = self.cal_MSE(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_mse += loss.item()
            train_mae += self.cal_MAE(outputs, targets).item()
        self.total_time_list.append(time.time() - start_time)
        avg_train_mse = train_mse / len(self.train_loader)
        avg_train_mae = train_mae / len(self.train_loader)
        self.train_mse_list.append(avg_train_mse)
        self.train_mae_list.append(avg_train_mae)
        return avg_train_mse, avg_train_mae

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
        self.test_mse_list.append(avg_test_mse)
        self.test_mae_list.append(avg_test_mae)
        return avg_test_mse, avg_test_mae

    # 存储计算的各个指标到
    def save_results(self):
        results = {}
        if os.path.exists(self.results_save_path):
            results = read_json(self.results_save_path)
        epoch_list = list(range(1, self.n_epoch + 1))
        # 单结点简记为sn，多结点简记为mn
        new_results = {
            "epoch_list": epoch_list,
            "sn_train_mse_list": self.train_mse_list,
            "sn_train_mae_list": self.train_mae_list,
            "sn_test_mse_list": self.test_mse_list,
            "sn_test_mae_list": self.test_mae_list,
            "sn_total_time_list": self.total_time_list,
        }
        for key, value in results.items():
            if key not in new_results:
                new_results[key] = value
        write_json(new_results, self.results_save_path)

    def out_message(self, message: str):
        logging.info(message)
        print(message)

    def save_model(self, model_name):
        save_path = os.path.join(self.model_save_dir, model_name)
        torch.save(self.model.state_dict(), save_path)

    def train(self):
        for epoch in range(1, self.n_epoch + 1):
            train_mse, train_mae = self.train_epoch(epoch)
            test_mse, test_mae = self.test_epoch(epoch)
            message = (
                f"Epoch: {epoch:>2}    TrainMSE: {train_mse:>.4f}   TrainMAE: {train_mae:>.4f}"
                + f"   TestMSE: {test_mse:>.4f}   TestMAE: {test_mae:>.4f}"
            )
            self.out_message(message)

            if epoch % self.model_save_epoch == 0:
                model_name = MODEL_NAME.format(epoch)
                self.save_model(model_name)
        self.save_results()


if __name__ == "__main__":
    builder = DatasetBuilder()
    trainset = builder.get_trainset()
    testset = builder.get_testset()

    trainer = Trainer(
        trainset=trainset,
        testset=testset,
        hidden_size=HIDDEN_SIZE,
    )

    trainer.train()
