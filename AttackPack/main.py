from .consts import *
from . import DATABASE as db, AttackEntry

def loop(model, loader, device, criterion, method, eps, type="d"):
    """
    A single loop over the dataset, while applying the attack method
    """

    for data, rr, target in loader:
        # Send the data and label to the device
        data, rr, target = data.to(device), rr.to(device), target.to(device)

        # todo needs requires_grad = True for attack?

        # Forward pass the data through the model
        output, ln = model(data, rr)

        # Get the initial prediction
        init_pred = target.argmax(dim=-1)

        # todo skip is init is already bad?
        if init_pred.item() != target.argmax(dim=-1).item():
            continue

        # todo? Optim step for model

        # Generate noise
        d_noise, rr_noise = method(data, rr, output, target, criterion, device)

        # Apply the noise to the data
        if "d" in type:
            # Add noise to the data
            data = data + eps * d_noise
            # todo? Normalize the data

        if "r" in type:
            # Add noise to the rr intervals
            rr = rr + eps * rr_noise


        # get the new output
        m_output, m_ln = model(data)

        # Get the new prediction
        mod_pred = m_output.argmax(dim=-1)

        # Check if the prediction changed
        if mod_pred.item() != init_pred.item():
            # If the prediction changed, we have a successful attack
            print(f"Attack successful: {init_pred.item()} -> {mod_pred.item()} with eps={eps}")
            # save to database
            entry = AttackEntry(
                model_name=model.__class__.__name__,
                weights_file=MODEL_WEIGHTS_PATH,
                epsilon=eps,
                attack_type=method.__name__,
                extent="both" if "d" in type and "r" in type else "data" if "d" in type else "rr",

            )


def main():
    pass
