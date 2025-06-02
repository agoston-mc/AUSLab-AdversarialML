import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np



def attack_model(*args, **kwargs):
    """
    Wrapper function to call the attack_model_grad function.
    """
    # Unpack arguments
    model, criterion, dev, loader, epsilon = args[:5]
    method = kwargs.get('method', 'fgsm')

    # Call the attack_model_grad function
    return attack_model_grad(model, criterion, dev, loader, epsilon, method)





def attack_model_grad(model, criterion, dev, loader, epsilon, method="fgsm"):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, rr, target in loader:

        # Send the data and label to the device
        # data, target = data.to(dev), target.to(dev)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output, ln = model(data, rr)

        lab = target.argmax(dim=-1)
        init_pred = target.argmax(dim=-1)
        # init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != lab.item():
            continue

        # Calculate the loss
        loss = criterion((output, ln), target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        # todo check?
        # data_denorm = denorm(data)

        # Call FGSM Attack
        # perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized, rr)

        # Check for success
        final_pred = output[0].argmax(dim=-1)

        if final_pred.item() == lab.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

