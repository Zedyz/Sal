DEVICE = 'cuda'
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4

IN_H, IN_W = 320, 480
OUT_H, OUT_W = 40, 60
GT_H, GT_W = 720, 1280

LAMBDA_NSS = 0.5
LAMBDA_KL = 1.0
LAMBDA_TV = 0.8

USE_FACE_MASK = True
USE_TEXT_MASK = True
USE_BANNER_MASK = True

ALPHA_FACE = 0.6
ALPHA_TEXT = 0.4
ALPHA_BANNER = -2.5

train_json = "../datasets/scanpaths_train.json"
val_json = "../datasets/scanpaths_test.json"

TEST_JSON_PATH = "../dataset/scanpaths_test.json"
ROOT_FOLDER = "."
CKPT_PATH = "best_fullres_tv.pth"
OUT_FOLDER = "out_infer_tv"
