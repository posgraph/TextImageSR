from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
#config.TRAIN.batch_size = 16
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.lr_text_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 200
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '/root/workspace/make_dataset3/input/full/'
config.TRAIN.lr_img_path = '/root/workspace/make_dataset3/input/full/'
config.TRAIN.hr_mask_img_path = '/root/workspace/make_dataset3/input/blank/'
config.TRAIN.hr_natural_img_path = '/root/workspace/make_dataset3/input/image/'


config.VALID = edict()
## test set location

config.VALID.hr_img_path = '/root/workspace/make_dataset2/text/'
config.VALID.lr_img_path = '/root/workspace/make_dataset2/text/'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
