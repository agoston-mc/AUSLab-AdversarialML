import os.path

import torch

from . import DATABASE as db, AttackEntry, MethodParams, SetupParams, bridge, methods
from dataclasses import dataclass


@dataclass
class effects:
    d_correct: float
    d_adv: float
    d_diff: float


def calc_attack_effectiveness(orig_pred, mod_pred):
    # assumes mod_pred is correctly a different idx than pred, a successful attack
    idx_orig = orig_pred.argmax(dim=-1)
    idx_mod = mod_pred.argmax(dim=-1)

    d_correct = orig_pred.flatten()[idx_orig] - mod_pred.flatten()[idx_orig]
    d_adv = mod_pred.flatten()[idx_mod] - orig_pred.flatten()[idx_mod]
    d_diff = mod_pred.flatten()[idx_mod] - mod_pred.flatten()[idx_orig]

    return effects(d_correct, d_adv, d_diff)


def loop(model, loader, device, criterion, method, epss, type="d", early_stop=-1, w_file="", data_file=None):
    """
    A single loop over the dataset, while applying the attack method
    """

    effs = []
    corrs = [0 for _ in epss]

    for data_idx, (data, rr, target) in enumerate(loader):

        if len(effs) >= early_stop > 0:
            break

        # Send the data and label to the device
        data, rr, target = data.to(device), rr.to(device), target.to(device)

        # Setup method
        method.setup(SetupParams(model, data, rr))

        # Forward pass the data through the model
        output, ln = model(data, rr)

        # Get the initial prediction
        init_pred = output.argmax(dim=-1)

        # Skip if initial prediction is wrong
        if init_pred.item() != target.argmax(dim=-1).item():
            continue

        # Compute loss and gradients
        loss = criterion((output, ln), target)
        loss.backward()

        for eps_idx, eps in enumerate(epss):  # Fixed variable shadowing
            # Generate noise with epsilon parameter
            mpar = MethodParams(data, rr, output, target, criterion, device, model, epsilon=eps)
            d_noise, rr_noise = method.get_noise(mpar)

            # Apply the noise based on attack type
            if "d" in type:
                adv_data = data + eps * d_noise
            else:
                adv_data = data

            if "r" in type:
                adv_rr = rr + eps * rr_noise
            else:
                adv_rr = rr

            # Get new output
            m_output, m_ln = model(adv_data, adv_rr)
            mod_pred = m_output.argmax(dim=-1)

            # Check if attack was successful
            if mod_pred.item() != init_pred.item():
                print(f"Attack successful on ds[{data_idx}]: {init_pred.item()} -> {mod_pred.item()} with eps={eps}")
                print(f"initial output: {output.flatten()}")
                print(f"modified output: {m_output.flatten()}")

                # Save to database
                entry = AttackEntry(
                    model_name=model.__class__.__name__,
                    weights_file=w_file,
                    epsilon=eps,
                    attack_type=method.name,
                    extent="both" if "d" in type and "r" in type else "data" if "d" in type else "rr",
                    data_idx=data_idx,
                    data_file=data_file,
                    data_noise=d_noise if "d" in type else None,
                    rr_noise=rr_noise if "r" in type else None
                )

                db.insert(entry)
                effs.append(calc_attack_effectiveness(output, m_output))
                corrs[eps_idx] -= 1

            corrs[eps_idx] += 1

    return effs, [corr / len(loader) for corr in corrs]


def load_module_from_weight(fweight: str, device):
    t_weights = torch.load(fweight, map_location=device, weights_only=True)

    if fweight.endswith(".pt"):
        fweight = fweight.strip(".pt")

    parts = fweight.split("_")

    module_name = parts[0]

    b_size = int(parts[1][1:])

    lr = float(parts[2][2:])

    epoch = int(parts[3][1:])

    p_init = [float(p) for p in parts[4][1:].split("-")]

    n_vp = int(parts[5][1:])

    n_hidden = [int(p) for p in parts[6][1:].split("-")]

    penalty = float(parts[7][1:])

    return [module_name, b_size, lr, epoch, p_init, n_vp, n_hidden, penalty], t_weights


def create_dataset(mfile, device, dtype=torch.float):
    """
    Create a dataset from a .mat file.
    """
    mfile = os.path.join(os.path.dirname(__file__), '..', 'vpunfold', 'data', mfile)
    dataset = bridge.create_dataset(mfile, device=device, dtype=torch.float)

    return dataset


def main(**kwargs):
    method: str = kwargs["method"]
    eps: list[float] = kwargs["eps"]
    device: str | torch.device = kwargs["device"]
    weight_f: str = kwargs["weights"]
    adv_method: str = kwargs["adv_method"]
    early_stop: int = kwargs.get("early_stop", -1)

    if method == "full":
        pass
    method = method.split(",")

    if isinstance(device, str):
        device = torch.device(device)


    m_fname = 'mitdb_filt35_w300adapt_ds2_float.mat'

    dataset = create_dataset(m_fname, device, dtype=torch.float)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    (name, *hypparams), weights = load_module_from_weight(weight_f, device)

    name, model, criterion = bridge.create_model(name,
                                                 hypparams,
                                                 dataset[0], torch.float, device)

    model.load_state_dict(weights)

    for m in method:
        print(f"Using method: {m}")
        effs = loop(
            model,
            loader,
            device,
            criterion,
            methods[m],
            epss=eps,
            type=adv_method,
            early_stop=early_stop,
            w_file=weight_f,
            data_file=m_fname
        )
        print(effs)


