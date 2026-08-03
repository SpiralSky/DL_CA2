Module src.models.autoencoders.VAE
==================================

Classes
-------

`VAE(encoder: torch.nn.modules.module.Module, decoder: torch.nn.modules.module.Module, latent_dim: int)`
:   Generic autoencoders shell: takes an encoder and decoder as injected nn.Module
    instances rather than hardcoding architecture. Swapping in a different
    encoder/decoder (deeper convs, ResNet blocks, conditional variants that
    also accept a label) requires no changes here.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `reparameterize(mu, logvar)`
    :

    ### Methods

    `forward(self, input_features) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `sample(self, num_samples: int, device=None)`
    :   Draws num_samples latent vectors from the prior N(0, I) and decodes
        them, for inspecting what the model has learned to generate.