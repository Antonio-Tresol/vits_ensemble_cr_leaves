# models
LR = 0.01
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4

# train loop
BATCH_SIZE = 64
TEST_SIZE = 0.5
TRAIN_SIZE = 1 - TEST_SIZE
EPOCHS = 5

# callback
PATIENCE = 3

# training loop
NUM_TRIALS = 2

# file paths
ROOT_DIR = "CRLeaves/"
INDICES_DIR = "Indices/"
CHECKPOINTS_DIR = "checkpoints/"
WANDB_PROJECT = "CR_Leaves"

# file paths
ROOT_DIR = "CRLeaves/"
INDICES_DIR = "Indices/"
CHECKPOINTS_DIR = "checkpoints/"
WANDB_PROJECT = "CR_Leaves"

# model directories
VIT_BASE_16_DIR = CHECKPOINTS_DIR + "vit_base_16/"
VIT_BASE_32_DIR = CHECKPOINTS_DIR + "vit_base_32/"
VIT_LARGE_32_DIR = CHECKPOINTS_DIR + "vit_large_32/"
DEIT3_BASE_16_DIR = CHECKPOINTS_DIR + "deit3_base_16/"
CONVNEXT_DIR = CHECKPOINTS_DIR + "convnext/"
RESNET_DIR = CHECKPOINTS_DIR + "resnet/"
EFFICIENTNET_DIR = CHECKPOINTS_DIR + "efficientnet/"

# model file names
VIT_BASE_16_FILENAME = "vit_base_16_"
VIT_BASE_32_FILENAME = "vit_base_32_"
VIT_LARGE_32_FILENAME = "vit_large_32_"
DEIT3_BASE_16_FILENAME = "deit3_base_16_"
CONVNEXT_FILENAME = "convnext_"
RESNET_FILENAME = "resnet_"
EFFICIENTNET_FILENAME = "efficientnet_"

# checkpoint parameters
TOP_K_SAVES = 1
