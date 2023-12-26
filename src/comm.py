import os
import time
import socket
import pickle
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Condition
from config import *
from utils import *


"""
该模型是为了实现分布式通信
Communicator类在初始化时会构造环形网络,保持每个结点与右侧结点的连接

allreduce的实现通过发送线程和接收线程实现而不是reducescatter+allgather

"""


# 环形结构的通信，每个结点只会把数据发送给右边结点并从左边结点接收数据
class Communicator:
    def __init__(self, rank: int = RANK, sync_flag: bytes = SYNC_FLAG):
        self.sync_flag = sync_flag
        self.n_hosts = len(HOSTS)
        # buckets_by_part[i][j]表示第j个bucket的第i个part的所有tensor
        self.buckets_by_part: list[list[list[Tensor]]] = None
        self.rank = rank
        # sock为当前结点的socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOSTS[self.rank], PORT))
        self.sock.listen(1)
        # left_conn存储该结点与左侧结点的连接，让连接持续保存避免重复建立
        self.left_conn = None
        # right_sock为该结点右侧结点的socket，一开始就直接连接，避免重复连接
        self.right_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pool = ThreadPoolExecutor(max_workers=2 * 3)
        # send_condition用于发送线程和接收线程之间的同步
        self.send_count = 1
        self.send_condition = Condition()
        self.counter = 0

        print()
        connect_right_future = self.pool.submit(self.connect_to_right)
        self.connect_to_left()
        connect_right_future.result()
        print()

    def right_rank(self, rank: int = None):
        if rank is None:
            rank = self.rank
        return (rank + 1) % self.n_hosts

    def left_rank(self, rank: int = None):
        if rank is None:
            rank = self.rank
        return (rank - 1 + self.n_hosts) % self.n_hosts

    def connect_to_right(self, port: int = PORT):
        print(f"--- Rank{self.rank} waiting for Rank{self.right_rank()} ---")
        host = HOSTS[self.right_rank()]
        connected = False
        max_retries = 1000
        retries = 0
        # 多次尝试直到连接成功
        while not connected and retries < max_retries:
            try:
                self.right_sock.connect((host, port))
                connected = True
            except socket.error as e:
                time.sleep(0.1)
                retries += 1
        # 连续多次仍然失败则抛出异常
        if not connected:
            raise ConnectionError(
                f"Unable to connect to {(host, port)} after {max_retries} retries."
            )
        print(
            f"--- Rank{self.rank} successfully connect to Rank{self.right_rank()} ---"
        )

    # 获取当前结点与左侧结点的连接，用于后续接收数据
    def connect_to_left(self):
        print(f"--- Rank{self.rank} waiting for Rank{self.left_rank()} ---")
        self.left_conn, _ = self.sock.accept()
        print(
            f"--- Rank{self.rank} successfully connect to Rank{self.left_rank()} ---")

    def send_data_to_right(self, part_id: int):
        data = pickle.dumps(self.buckets_by_part[part_id])
        data_size = len(data).to_bytes(4, byteorder="big")
        self.right_sock.sendall(data_size)
        self.right_sock.sendall(data)
        # print(f"--Rank{self.rank} send {part_id}th data to Rank{self.right_rank()}")

    def recv_data_from_left(self, part_id: int, add=False):
        # print(f"--Rank{self.rank} waiting to receive {part_id}th data of Rank{self.left_rank()}")
        def add_all_tensor(x: list[list[Tensor]], y: list[list[Tensor]]):
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] += y[i][j]

        data_size = int.from_bytes(
            recv_all(self.left_conn, 4), byteorder="big")
        data = recv_all(self.left_conn, data_size)
        data: list[list[Tensor]] = pickle.loads(data)
        if add:
            add_all_tensor(self.buckets_by_part[part_id], data)
        else:
            self.buckets_by_part[part_id] = data
        # print(f"--Rank{self.rank} receive {part_id}th data of Rank{self.left_rank()}")

    # 发送过程,每一轮发送后part_id左移
    def send_process(self, beg_part_id: int):
        part_id = beg_part_id
        n_epoch = (self.n_hosts - 1) * 2
        for i in range(n_epoch):
            with self.send_condition:
                # 当send_count小于等于0时，线程等待
                while self.send_count <= 0:
                    self.send_condition.wait()
                self.send_count -= 1
                self.send_data_to_right(part_id)
            part_id = self.left_rank(part_id)

    # 发送过程,每一轮接收后part_id左移,迭代一半后进行allgather,不需要再累加
    def recv_process(self, beg_part_id: int):
        part_id = beg_part_id
        n_epoch = (self.n_hosts - 1) * 2
        for i in range(n_epoch):
            is_reducescatter = i < n_epoch // 2
            self.recv_data_from_left(part_id, add=is_reducescatter)
            
            with self.send_condition:
                self.send_count += 1
                self.send_condition.notify()

            part_id = self.left_rank(part_id)

    # 给右侧结点发送同步信号
    def send_sync_signal_to_right(self):
        self.right_sock.sendall(self.sync_flag)

    # 等待接收来自左侧节点的同步信号
    def wait_sync_signal_from_left(self):
        recv_all(self.left_conn, len(self.sync_flag))
    
    # 确保所有结点都同时执行allreduce
    def barrier(self):
        sync_send_future = self.pool.submit(self.send_sync_signal_to_right)
        self.wait_sync_signal_from_left()
        sync_send_future.result()

    # 在已经建立好的三台主机的环形结构中通过allreduce同步数据
    # 同步的单位是buckets
    def allreduce(self, buckets: list[list[Tensor]]) -> list[list[Tensor]]:
        shapes_by_bucket = flatten(buckets)
        self.buckets_by_part, pre_tensor_id = split_buckets(
            buckets, num_parts=self.n_hosts
        )

        self.barrier()
        beg_send_part_id = self.rank
        beg_recv_part_id = self.left_rank(beg_send_part_id)
        send_future = self.pool.submit(self.send_process, beg_send_part_id)
        recv_future = self.pool.submit(self.recv_process, beg_recv_part_id)
        send_future.result()
        recv_future.result()

        buckets = merge_buckets(self.buckets_by_part, pre_tensor_id)
        unflatten(buckets, shapes_by_bucket)

        # 平均聚合后的值
        for i in range(len(buckets)):
            for j in range(len(buckets[i])):
                buckets[i][j] = buckets[i][j] / self.n_hosts

        self.counter += 1

        return buckets

    # 实现对数据的分布式采样，返回值是当前Node的数据
    def sample_data(self, dataset: Dataset, seed=42):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)
        split_parts_size = split(len(indices), self.n_hosts)
        split_parts_size.reverse()  # 让第一个结点分配更少的数据
        indices = indices.split(split_parts_size, dim=0)[self.rank]
        return Subset(dataset, indices=indices)

    def close(self):
        self.sock.close()
        self.left_conn.close()
        self.right_sock.close()
        self.pool.shutdown()


if __name__ == "__main__":
    print(f"Test allreduce ----------------------------------------------------")
    data = [
        [torch.arange(16).reshape(4, 4)],
        [torch.arange(5), torch.arange(6)],
    ]
    print("\nInitial data: ")
    print(data)

    communicator = Communicator()
    data = communicator.allreduce(data)
    print("\nFinally data: ")
    print(data)
    print()

    print(f"Test sample_data ----------------------------------------------------")

    class MyDataset(Dataset):
        def __init__(self, n):
            super().__init__()
            self.data = list(range(n))

        def __getitem__(self, i):
            return self.data[i]
        
        def __len__(self):
            return len(self.data)

        def print(self):
            print(self.data)
    def get_data(dataset):
        return [dataset[i] for i in range(len(dataset))]
    dataset = MyDataset(10)
    print("Full data:")
    print(get_data(dataset))
    print()
    subset = communicator.sample_data(dataset)
    print("Part data:")
    print(get_data(subset))
    print()

    communicator.close()
