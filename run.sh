
(
    bash ./src/train_sn.sh;
    echo Single-Node-Training is finished.
) &

(
    bash ./src/train_mn.sh;
    echo Multi-Node-Training is finished.
) &

wait

python ./src/draw.py

