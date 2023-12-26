#!/bin/bash

# 该脚本是为了启动多个结点的分布式训练
# 特别要注意多个结点的环境需要一致，最好处在配置相同的conda环境中

# 首先将文件发给另外的结点
python ./src/send_to_all_hosts.py

# 然后在每个主机的目录下启动程序,并设置将程序的输出重定向到当前工作目录下
#   输出文件命名为 {RANK}_{HOST}_stdout.txt

target_dir="/data/lvqs/ToyDDP"
src_file="./src/train_mn.py"

HOSTS=("10.103.10.158" "10.103.10.157" "10.103.11.151")
# 每台主机的base环境激活脚本的路径
ACTIVATE_PATHS=("/data/lvqs/anaconda3/bin/activate" "/data/lvqs/anaconda3/bin/activate" "/data/lvqs/miniconda3/bin/activate")
USER="lvqs"


for ((RANK=0;RANK<${#HOSTS[@]};RANK++))
do
    # 先激活base环境再运行
    ( 
        HOST=${HOSTS[$RANK]} ; \
        ACTIVATE_PATH=${ACTIVATE_PATHS[$RANK]} ; \

        ssh $USER@$HOST "source $ACTIVATE_PATH base && cd $target_dir && \
            python -u $src_file" > "stdout_$RANK"_"$HOST.txt" 2>&1 

        # ssh $USER@$HOST "source $ACTIVATE_PATH base && python -V"
    ) &
done


wait
