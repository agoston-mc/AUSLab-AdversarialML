# path hack to allow for importing vpunfold methods
import os
import sys
# add ../vpunfold to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vpunfold')))

import vpunfold.vpnet as vpnet
import vpunfold.ecg_classification as ecg_classification
import vpunfold.ecg_tools as ecg_tools
import torch

def create_dataset(file, device, dtype) -> torch.utils.data.Dataset:
    return ecg_tools.ecg_dataset(file, device=device, dtype=dtype)

def create_model(name, hyperparams, sample_entry, dtype, device) -> tuple[str, vpnet.VPNet, vpnet.VPLoss]:
    """
    Helper function to force fill scrip local variables in ecg_classification and calculate some sizes for agglomeration
    """
    # get creator function
    assert name in [
        "vpnet_hermite", "vpnet_hermite_rr", "vpnet_hermite2", "vpnet_hermite2_rr",
    ], f"Unknown model name (unfold fails): {name}"
    builder = getattr(ecg_classification, name)

    # get hyperparams
    batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty = hyperparams

    # overwrite ecg_classification variables (uggo prepare)
    sample, rr, label = sample_entry
    n_channels, n_in = sample.shape
    _, n_rr = rr.shape
    n_out, = label.shape

    # eww
    # i'm sorry
    ecg_classification.__dict__.update({
        'n_channels': n_channels, 'n_in': n_in, 'n_rr': n_rr, 'n_out': n_out,

        "device": device, "dtype": dtype
    })

    # create model
    name, model, criterion = builder(batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty)

    return name, model, criterion