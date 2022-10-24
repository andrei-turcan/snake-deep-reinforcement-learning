import cv2

RANDOM_SEED = 42

ENV_CONFIG = {
  'show': False,
  'render_delay': 0,
  'grid_size': 12,
  'observation_shape': (96, 96, 1),
  'observation_resize_interpolation': cv2.INTER_AREA,
  'apple_reward': 1,
  'defeat_reward': -1,
  'move_reward': 0,
}

TRAIN_CONFIG = {
  'num_gpus': 0,
  'num_workers': 1,
  'max_steps': 2000000,
  'train_batch_size': 32,
  'checkpoint_freq': 50
}
