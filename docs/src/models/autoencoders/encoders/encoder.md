Module src.models.autoencoders.encoders.encoder
===============================================

Classes
-------

`BasicEncoder(input_channels=3, output_channels=32, latent_dim=128)`
:   Convolutional encoder for 32x32 RGB images (e.g. CIFAR-10).
    Maps an image to the parameters (mu, logvar) of a diagonal Gaussian
    over the latent space. Downsamples 32 -> 16 -> 8 -> 4 via stride-2 convs,
    with a stride-1 refinement conv at each resolution so the network has
    capacity to learn shape/structure features before compressing further,
    rather than immediately squeezing spatial detail into the bottleneck.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, inputs) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.