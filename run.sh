
# 串行执行单/多结点
bash ./src/train_mn.sh
bash ./src/train_sn.sh

# # 并行执行单/多结点
# bash ./src/train_sn.sh &
# bash ./src/train_mn.sh &

wait
python ./src/draw.py

