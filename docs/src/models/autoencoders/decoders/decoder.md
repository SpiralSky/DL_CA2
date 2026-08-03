Module src.models.autoencoders.decoders.decoder
===============================================

Classes
-------

`BasicDecoder(out_channels=3, base_channels=32, latent_dim=128)`
:   Convolutional decoder mirroring ConvEncoder. Maps a latent vector back
    to a 32x32 RGB image via upsampling transposed convs: 4 -> 8 -> 16 -> 32,
    with a stride-1 refinement conv at each resolution (mirroring the
    encoder) so shape/structure can be reconstructed with more capacity than
    a single upsampling conv provides. Output is passed through sigmoid to
    match [0, 1]-scaled ToTensor() inputs.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, input_features) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.