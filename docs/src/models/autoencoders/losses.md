Module src.models.autoencoders.losses
=====================================

Functions
---------

`kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) ‑> torch.Tensor`
:   

`reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, loss_type: Literal['mse', 'bce']) ‑> torch.Tensor`
:   

`vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, *, beta: float = 1.0, recon_loss_type: Literal['mse', 'bce'] = 'mse', free_bits: float = 0.0) ‑> src.models.autoencoders.losses.VAELossOutput`
:   

Classes
-------

`VAELossOutput(*args, **kwargs)`
:   dict() -> new empty dictionary
    dict(mapping) -> new dictionary initialized from a mapping object's
        (key, value) pairs
    dict(iterable) -> new dictionary initialized as if via:
        d = {}
        for k, v in iterable:
            d[k] = v
    dict(**kwargs) -> new dictionary initialized with the name=value pairs
        in the keyword argument list.  For example:  dict(one=1, two=2)

    ### Ancestors (in MRO)

    * builtins.dict

    ### Class variables

    `kl_divergence: torch.Tensor`
    :

    `reconstruction: torch.Tensor`
    :

    `total: torch.Tensor`
    :