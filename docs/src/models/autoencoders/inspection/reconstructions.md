Module src.models.autoencoders.inspection.reconstructions
=========================================================

Functions
---------

`plot_reconstructions(model, data_loader, device=None, num_images=8)`
:   Draws one batch from data_loader, runs it through the model, and plots
    original vs. reconstructed images side by side (originals on top row,
    reconstructions on bottom row). This catches issues loss curves alone
    can hide, e.g. whether the model has collapsed to near-identical blurry
    outputs regardless of input.