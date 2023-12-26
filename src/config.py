
SYNC_FLAG = b"<SYNC>"  # 用于同步所有结点到allreduce前
HOSTS = ["10.103.10.158", "10.103.10.157", "10.103.11.151"]
USER = "lvqs"
PORT = 26001   # 通信时使用的端口号
RANK = 0   # 该结点的rank

INPUT_SIZE = 1
INPUT_LEN = 120
OUTPUT_LEN = 12

PATIENT_ID = 100   # 绘制曲线的病人ID
NUM_PREDICTION_POINTS = 300

IMAGE_DIR = "./image"
INITIAL_MODEL_PATH = "./data/initial_model.pt"
MODEL_NAME = "epoch_{}.pt"

HIDDEN_SIZE = 200
NUM_LAYERS = 2

DRAW_EPOCH = 50

n_epoch_sn = 50
lr_sn = 1e-2
lr_step_size_sn = 10
lr_gamma_sn = 0.7
batch_size_sn = 3072

n_epoch_mn = 50
lr_mn = 1e-2
lr_step_size_mn = 10
lr_gamma_mn = 0.7
batch_size_mn = 1024

BUCKET_SIZE = int(5e6)
