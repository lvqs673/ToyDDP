
SYNC_FLAG = b"<SYNC>"  # 用于同步所有结点到allreduce前
HOSTS = ["10.103.10.158", "10.103.10.157", "10.103.11.151"]
USER = "lvqs"
PORT = 26001   # 通信时使用的端口号
RANK = 0   # 该结点的rank

INPUT_SIZE = 1
INPUT_LEN = 120
OUTPUT_LEN = 12

PATIENT_ID = 100   # 绘制曲线的病人ID
NUM_PREDICTION_POINTS = 100

IMAGE_DIR = "./image"
MODEL_NAME = "epoch_{}.pt"

HIDDEN_SIZE = 200
NUM_LAYERS = 2


n_epoch_sn = 200
lr_sn = 5e-3
batch_size_sn = 5120

n_epoch_mn = 200
lr_mn = 5e-3
batch_size_mn = 5120

BUCKET_SIZE = int(5e6)
