Module src.models.autoencoders.model_factory
============================================

Functions
---------

`basic_autoencoder(in_channels=3, base_channels=32, latent_dim=128)`
:   Assembles the baseline autoencoders from its component modules. Switching to a
    different encoder/decoder implementation later only means changing what
    gets constructed here (or adding an entry to MODEL_REGISTRY below) --
    everything downstream (training loop, loss function) is unaffected.