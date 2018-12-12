# Final Project:

## Draft

Key Reference: https://worldmodels.github.io/


## Code:
- model
   - mdnrnn.py: contains pytorch model and loss for M
   - vaelin.py: contains pytorch model and losses for the VAE (dense)
   - vae.py: contains pytorch model and losses for the VAE (convolutional)
- dataset.py: dataloaders for vae and rnn training generated from rollouts
- generate_data.py: script to generate rollouts from the environment
- resample_vae.py: resampling scheme for making data distribution flatter while training VAE
- train_x.py: scripts to train x
