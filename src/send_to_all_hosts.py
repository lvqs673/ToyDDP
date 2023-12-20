import os
from config import *
from concurrent.futures import ThreadPoolExecutor

"""

本文件的目的是为了把数据和代码发给所有用于分布式训练的主机
在分布式训练之前启动，确保所有主机都有源代码和数据文件

默认建立在每个主机的"/data/lvqs/"目录下

"""


dest_project_dir = "/data/lvqs/ToyDDP/"
dest_data_dir = os.path.join(dest_project_dir, "data")
dest_glucose_dir = os.path.join(dest_data_dir, "blood_glucose")
dest_src_dir = os.path.join(dest_project_dir, "src")
dest_config_file = os.path.join(dest_src_dir, "config.py")

data_dir = "./data/"
glucose_dir = "./data/blood_glucose/"
src_dir = "./src/"


def send_file(args: tuple[int, str]):
    rank, host = args
    os.system(f"ssh {USER}@{host} 'mkdir -p {dest_project_dir}'")
    os.system(f"ssh {USER}@{host} 'mkdir -p {dest_data_dir}'")
    os.system(f"ssh {USER}@{host} 'mkdir -p {dest_glucose_dir}'")
    os.system(f"ssh {USER}@{host} 'mkdir -p {dest_src_dir}'")

    os.system(f"rsync -az {glucose_dir} {USER}@{host}:{dest_glucose_dir}")
    exclude_files = ["__pycache__"]   # 不发送python缓存
    exclude_options = " ".join([f"--exclude={file}" for file in exclude_files])
    os.system(f"rsync -az {exclude_options} {src_dir} {USER}@{host}:{dest_src_dir}")
    # 修改其它结点的 Rank
    os.system(
        f"ssh {USER}@{host} \"sed -i 's/RANK = 0/RANK = {rank}/g' {dest_config_file}\""
    )

    print(f"Rank{rank} is ready.")


n_hosts = len(HOSTS)
params = enumerate(HOSTS[1:], 1)


with ThreadPoolExecutor(n_hosts - 1) as pool:
    res = pool.map(send_file, params)
    list(res)
