

HOSTS = ["10.103.10.158", "10.103.10.157", "10.103.11.151"]
USER = "lvqs"
PORT = 26001   # 通信时使用的端口号
RANK = 0   # 该结点的rank

INPUT_SIZE = 1
INPUT_LEN = 60
OUTPUT_LEN = 6

PATIENT_ID = 100   # 绘制曲线的病人ID
NUM_PREDICTION_POINTS = 100

MODEL_NAME = "epoch_{}.pt"

HIDDEN_SIZE = 50
NUM_LAYERS = 2


n_epoch_sn = 200
lr_sn = 1e-3
batch_size_sn = 64
