import cv2

RANDOM_SEED = 42

MAX_STEPS = 100000

CHECKPOINT_FREQ = 100

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

CONFIG = {
  'env': 'snake_env',
  'num_gpus': 1,
  'num_workers': 1,
  'framework': 'torch',
  'seed': RANDOM_SEED,
  'env_config': ENV_CONFIG,
  'model': {
    'conv_filters': [
      [16, 8, 4],
      [32, 4, 2],
      [64, 3, 1],
      [64, 3, 1],
      [64, 3, 1],
      [64, 3, 1],
      [64, 3, 1],
      [128, 12, 1],
    ],
    'fcnet_hiddens': [128]
  },
  'hiddens': [128],
  'train_batch_size': 32
}
