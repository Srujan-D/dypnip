BATCH_SIZE = 32 #8
INPUT_DIM = 4
EMBEDDING_DIM = 128
SAMPLE_SIZE = 200
K_SIZE = 20
BUDGET_RANGE = (7,9)
SAMPLE_LENGTH = 0.2

ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4

USE_GPU = True
USE_GPU_GLOBAL = True
CUDA_DEVICE = [1]
NUM_META_AGENT = 32 #32 #6
LR = 3e-5
GAMMA = 0.99
DECAY_STEP = 32
SUMMARY_WINDOW = 8
FOLDER_NAME =  'weather_model_pr_wtr_small_new_norm'#'robust_vae_dim16_3fires' #'weather_model' #'vae_predict_next_belief_5fires' #'vae_predict_next_belief' #'robust_reverse_curriculum_vae_dim16' #'robust_lambda100_fuel1_5_10'
pretrain_model_path = f'model/robust_vae_dim16_3fires'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False
SAVE_IMG_GAP = 250

FIXED_ENV= 2

HISTORY_SIZE = (50, 101)
TARGET_SIZE = 1
HISTORY_STRIDE = 5
EPISODE_STEPS = 256

#GAE --> taken from STAMP
GAE_LAMBDA = 0

run_name = FOLDER_NAME
BUFFER_SIZE = int(NUM_META_AGENT * EPISODE_STEPS)

use_wandb = True

UPDATE_EPOCHS = 8

BELIEF_EMBEDDING_DIM = 16

SAVE_FEATURES = False