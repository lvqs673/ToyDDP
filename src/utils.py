import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence
from torch import Tensor
from config import *


def mean(lst: list):
    return sum(lst) / len(lst)

def split(n: int, k: int) -> list[int]:
    c, r = divmod(n, k)
    return [c + (i < r) for i in range(k)]


def read_json(file_path: str) -> object:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: object, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


# 确保sock刚好接收data_size字节的数据
def recv_all(sock, data_size: int) -> bytes:
    data = b""
    while len(data) < data_size:
        packet = sock.recv(data_size - len(data))
        if not packet:
            raise ConnectionError(f"Canot receive {data_size} byte data!")
        data += packet
    return data


# 分割模型梯度为多个指定大小的桶
def make_buckets(model: nn.Module, bucket_size=BUCKET_SIZE) -> list[list[Tensor]]:
    buckets = []
    current_bucket = []
    current_size = 0

    for param in model.parameters():
        if param.grad is not None:
            grad_size = param.grad.data.numel()
            if current_size + grad_size > bucket_size and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(param.grad)
            current_size += grad_size

    if current_bucket:
        buckets.append(current_bucket)

    return buckets


# 用buckets中的梯度修改模型梯度
def change_grad(model: nn.Module, buckets: list[list[Tensor]]):
    bucket_idx = 0  # 当前桶的索引
    grad_idx = 0  # 当前桶内梯度的索引

    for param in model.parameters():
        if param.grad is not None:
            param.grad = buckets[bucket_idx][grad_idx]
            grad_idx += 1

            # 如果当前桶内的梯度已经用完，移动到下一个桶
            if grad_idx >= len(buckets[bucket_idx]):
                bucket_idx += 1
                grad_idx = 0


# 将bucket中的每个tensor展平（原地操作）
# 返回shapes_by_bucket
# shapes_by_bucket[i]表示第i个bucket中每个tensor的shape
def flatten(buckets: list[list[Tensor]]) -> list[list[Tensor]]:
    shapes_by_bucket = []
    for bucket in buckets:
        shapes = []
        for i, _tensor in enumerate(bucket):
            shapes.append(_tensor.shape)
            bucket[i] = _tensor.view(-1)
        shapes_by_bucket.append(shapes)
    return shapes_by_bucket


# 将buckets按tensor的shape恢复到原来的tensor
def unflatten(buckets: list[list[Tensor]], shapes_by_bucket: list[list[Tensor]]):
    for bucket, shapes in zip(buckets, shapes_by_bucket):
        for pre_tensor_id, shape in enumerate(shapes):
            bucket[pre_tensor_id] = bucket[pre_tensor_id].view(shape)


# 将每个bucket分为num_splits个部分
# 返回值是buckets_by_part (list[list[list[Tensor]]])和tensor_id (list[list[list[int]]])
# buckets_by_part[i][j]是第j个bucket的第i个part的所有tensor
# pre_tensor_id[i][j][k]表示第j个bucket的第i个part的第k个tensor在这个bucket中的索引，用来恢复原来的buckets
#   当pre_tensor_idd[i][j][k1]==pre_tensor_id[i][j][k2]时，两个tensor需要合并
def split_buckets(
    buckets: list[list[Tensor]], num_parts: int
) -> tuple[list[list[list[Tensor]]], list[list[list[int]]]]:
    buckets_by_part = [[[] for _ in range(len(buckets))] for _ in range(num_parts)]
    pre_tensor_id = [[[] for _ in range(len(buckets))] for _ in range(num_parts)]
    for bucket_id, bucket in enumerate(buckets):
        n_total_params = sum(tensor.numel() for tensor in bucket)
        max_size_by_part = split(n_total_params, num_parts)
        _tensor_id = 0  # 当前tensor在bucket中的索引

        # max_size表示该部分应该分到的tensor元素个数
        for part_id, max_part_size in enumerate(max_size_by_part):
            cur_part_size = 0
            tensors = []  # 该part可以分到的tensor
            _tensor_ids = []  # 该part的每个tensor对应的在原bucket中的id
            while _tensor_id < len(bucket) and cur_part_size < max_part_size:
                tensor = bucket[_tensor_id]
                # n_valid_size表示实际可以分给当前部分的tensor元素个数，为了应对tensor在内部分割给两个部分的情况
                n_valid_size = min(tensor.numel(), max_part_size - cur_part_size)
                tensors.append(tensor[:n_valid_size])
                _tensor_ids.append(_tensor_id)
                cur_part_size += n_valid_size
                if n_valid_size == tensor.numel():
                    _tensor_id += 1
                else:
                    # 更新 tensor 为剩余部分
                    bucket[_tensor_id] = tensor[n_valid_size:]
            buckets_by_part[part_id][bucket_id] = tensors
            pre_tensor_id[part_id][bucket_id] = _tensor_ids

    return buckets_by_part, pre_tensor_id


# 合并到原来的buckets
def merge_buckets(
    buckets_by_part: list[list[list[Tensor]]], pre_tensor_id: list[list[list[int]]]
) -> list[list[Tensor]]:
    n_buckets = len(buckets_by_part[0])
    pre_buckets = []
    for bucket_id in range(n_buckets):
        tensors = []
        _pre_tensor_id = -1  # 记录上一个加入tensors的tensor在原bucket中的索引，用于合并被分割到两个parts的tensor
        for part_id, (buckets, _tensor_ids) in enumerate(
            zip(buckets_by_part, pre_tensor_id)
        ):
            # bucket为第bucket_id个桶的第part_id部分
            bucket = buckets[bucket_id]
            # tensor_ids为bucket中的每个tensor的在原bucket中的索引
            tensor_ids = _tensor_ids[bucket_id]
            for tensor, tensor_id in zip(bucket, tensor_ids):
                if tensor_id == _pre_tensor_id:
                    tensors[-1] = torch.cat([tensors[-1], tensor])
                else:
                    tensors.append(tensor)
                _pre_tensor_id = tensor_id
        pre_buckets.append(tensors)
    return pre_buckets


if __name__ == "__main__":
    # 测试分桶
    buckets = [
        [torch.arange(16).reshape(4, 4)],
        [torch.arange(5), torch.arange(6)],
    ]
    print("Initial buckets:")
    print(buckets)
    print()

    # Flatten
    shapes_by_bucket = flatten(buckets)
    print("Flattened buckets:")
    print(buckets)
    print("shapes by bucket:")
    print(shapes_by_bucket)
    print()

    buckets_by_part, pre_tensor_id = split_buckets(buckets, num_parts=3)
    print("Splitted buckets (buckets_by_part):")
    for part_id, buckets in enumerate(buckets_by_part):
        print(f"Part {part_id}: {buckets}")
    print("pre tensor_id:")
    print(pre_tensor_id)
    print()

    buckets = merge_buckets(buckets_by_part, pre_tensor_id)
    print("Merged buckets:")
    print(buckets)
    print()

    unflatten(buckets, shapes_by_bucket)
    print("Unflattened buckets:")
    print(buckets)
    print()
