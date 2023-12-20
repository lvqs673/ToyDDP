import os
import numpy as np
from numpy import ndarray
import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader,Subset
from config import *


class GlucoseDataset(Dataset):
    def __init__(self, data: Tensor | ndarray):
        super().__init__()
        self.data = torch.tensor(data)

    def __getitem__(self, i) -> tuple[Tensor, Tensor]:
        return self.data[i, :INPUT_LEN], self.data[i, INPUT_LEN:]

    def __len__(self):
        return len(self.data)


class DatasetBuilder:
    def __init__(self, data_dir: str = "./data/blood_glucose",
                 train_ratio=0.7):
        self.all_seqs = []  # 记录每个病人的血糖序列
        self.all_tables = []  # 记录每个病人的血糖序列转化为的数据
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            seq = pd.read_csv(filepath).iloc[:, 1].to_numpy(dtype=np.float32)
            table = self.seq2table(seq)
            self.all_seqs.append(seq)
            self.all_tables.append(table)
        self.table = np.concatenate(self.all_tables, axis=0)
        train_size = int(len(self.table)*train_ratio)
        self.train_table = self.table[:train_size]
        self.test_table = self.table[train_size:]

    def seq2table(self, seq: ndarray) -> None | ndarray:
        n = len(seq)
        k = INPUT_LEN+OUTPUT_LEN
        if k > n:
            return None
        table = np.zeros((n - k + 1, k), dtype=seq.dtype)
        for i in range(n - k + 1):
            table[i] = seq[i:i + k]
        return table

    def get_trainset(self,) -> GlucoseDataset:
        return GlucoseDataset(self.train_table)

    def get_testset(self,) -> GlucoseDataset:
        return GlucoseDataset(self.test_table)
    
    # 用来对100号患者的血糖序列进行预测
    def get_kth_dataset(self, k:int, size:int=-1) -> tuple[Dataset, list]:
        dataset = GlucoseDataset(self.all_tables[k-1])
        seq = self.all_seqs[k-1][INPUT_LEN:][:-OUTPUT_LEN+1].tolist()
        if size == -1:
            size = len(seq)
        return Subset(dataset, np.arange(size)), seq[:size]


if __name__ == "__main__":
    builder = DatasetBuilder()
    trainset = builder.get_trainset()
    testset = builder.get_testset()
    print("Trainset:")
    print(len(trainset))
    print(trainset[0])

    print("Testset:")
    print(len(testset))
    print(testset[0])

    print()
    dataset, seq=builder.get_kth_dataset(100)
    print("100th Dataset:")
    print(len(dataset))
    print(len(seq))
    print(dataset[0])
    print(seq[:OUTPUT_LEN])
