
# vae data generation
HEIGHT = 64
WIDTH = 64
NUM_EPISODES = 100
NUM_STEPS = 1000
ROLLOUT_DIR = './rollouts/'

# vae data resampling
VAE_FULL_DATASET_SIZE = 100000
VAE_USE_DATASET_SIZE = 10000

#vae training
VAE_PATH = './checkpoints/vae.tar'
LATENT_SIZE = 32
VAE_BATCH_SIZE = 32
VAE_EPOCHS = 50

# rnn training
RNN_DATASET_SIZE = 10
RNN_SAMPLE_SIZE = 100

