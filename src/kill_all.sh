#!/bin/bash

# 该脚本是为了杀死几个结点的进程

target_dir="/data/lvqs/ToyDDP"
src_file="./src/comm.py"

HOSTS=("10.103.10.158" "10.103.10.157" "10.103.11.151")

USER="lvqs"


for ((RANK=0;RANK<${#HOSTS[@]};RANK++))
do
    ( 
        HOST=${HOSTS[$RANK]} ; \
        ssh $USER@$HOST "pkill -f python.*src/train_mn.py$"
    )&
done


wait
