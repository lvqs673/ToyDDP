import os
import time
import socket
import pickle
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Event
from config import *
from utils import *


"""
该模型是为了实现分布式通信
Communicator类在初始化时会构造环形网络,保持每个结点与右侧结点的连接

allreduce的实现通过发送线程和接收线程实现而不是reducescatter+allgather

"""


# 环形结构的通信，每个结点只会把数据发送给右边结点并从左边结点接收数据
class Communicator:
    def __init__(self, rank: int = RANK, seed: int = 42):
        self.seed = seed
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
        # ready_to_send用于发送线程和接收线程之间的同步
        self.ready_to_send = Event()

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
        print(f"--- Rank{self.rank} successfully connect to Rank{self.left_rank()} ---")

    def send_data_to_right(self, part_id: int):
        data = pickle.dumps(self.buckets_by_part[part_id])
        data_size = len(data).to_bytes(4, byteorder="big")
        self.right_sock.sendall(data_size)
        self.right_sock.sendall(data)
        # print(f"Rank{self.rank} send {part_id}th data to Rank{self.right_rank()}")

    def recv_data_from_left(self, part_id: int, add=False):
        def add_all_tensor(x: list[list[Tensor]], y: list[list[Tensor]]):
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] += y[i][j]

        data_size = int.from_bytes(recv_all(self.left_conn, 4), byteorder="big")
        data = recv_all(self.left_conn, data_size)
        data: list[list[Tensor]] = pickle.loads(data)
        if add:
            add_all_tensor(self.buckets_by_part[part_id], data)
        else:
            self.buckets_by_part[part_id] = data
        # print(f"Rank{self.rank} receive {part_id}th data of Rank{self.left_rank()}")

    # 发送过程,每一轮发送后part_id左移
    def send_process(self, beg_part_id: int):
        part_id = beg_part_id
        n_epoch = (self.n_hosts - 1) * 2
        for i in range(n_epoch):
            self.ready_to_send.wait()
            self.send_data_to_right(part_id)
            self.ready_to_send.clear()
            part_id = self.left_rank(part_id)

    # 发送过程,每一轮接收后part_id左移,迭代一半后进行allgather,不需要再累加
    def recv_process(self, beg_part_id: int):
        part_id = beg_part_id
        n_epoch = (self.n_hosts - 1) * 2
        for i in range(n_epoch):
            is_reducescatter = i < n_epoch // 2
            self.recv_data_from_left(part_id, add=is_reducescatter)
            self.ready_to_send.set()
            part_id = self.left_rank(part_id)

    # 在已经建立好的三台主机的环形结构中通过allreduce同步数据
    # 同步的单位是buckets
    def allreduce(self, buckets: list[list[Tensor]]) -> list[list[Tensor]]:
        shapes_by_bucket = flatten(buckets)
        self.buckets_by_part, pre_tensor_id = split_buckets(
            buckets, num_parts=self.n_hosts
        )

        self.ready_to_send.set()
        beg_send_part_id = self.rank
        beg_recv_part_id = self.left_rank(beg_send_part_id)
        send_future = self.pool.submit(self.send_process, beg_send_part_id)
        recv_future = self.pool.submit(self.recv_process, beg_recv_part_id)
        send_future.result()
        recv_future.result()

        buckets = merge_buckets(self.buckets_by_part, pre_tensor_id)
        unflatten(buckets, shapes_by_bucket)
        return buckets

    # 实现对数据的分发，返回值是当前Node的数据
    def despatch_data(self, dataset: Dataset):
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(dataset), generator=generator)
        split_parts_size = split(len(indices), self.n_hosts)
        indices = indices.split(split_parts_size, dim=0)[self.rank]
        return Subset(dataset, indices=indices)

    def close(self):
        self.sock.close()
        self.left_conn.close()
        self.right_sock.close()
        self.pool.shutdown()


if __name__ == "__main__":
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

    communicator.close()
