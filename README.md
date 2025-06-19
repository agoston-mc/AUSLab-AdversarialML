# AUSLab-AdversarialML

Adversarial attack testing for the VPNet model.

## Overview

This project provides a framework for evaluating the robustness of the VPNet model against adversarial attacks. It implements various attack methods and measures their effectiveness on ECG data.

## Installation
 required dependencies:
   ``
   pip install -r requirements.txt
   ``

## Usage

### Running Attacks

You can run adversarial attacks using the provided Python interface. The main entry point is the `AttackPack.main()` function, which accepts several parameters to configure the attack.

#### Example

```python
import AttackPack

AttackPack.main(
    method="fgsm",                # Attack method name (e.g., "fgsm", "pgd", etc.)
    eps=[0.1, 0.01, 0.001],       # List of epsilon values for the attack
    device="cpu",                 # Device to run the attack on ("cpu" or "cuda")
    weights="model_weights.pt",   # Path to the model weights file
    adv_method="d",               # Attack type: "d" (data), "r" (rr), or "dr" (both)
    early_stop=10                 # (Optional) Stop after N successful attacks
)
```

##### Parameters

- `method`: Name of the attack method to use. Multiple methods can be specified as a comma-separated string.
- `eps`: List of epsilon values (perturbation strengths) to test.
- `device`: Device to run the computations on (`"cpu"` or `"cuda"`).
- `weights`: Path to the trained VPNet model weights file.
- `adv_method`: Type of adversarial attack:
    - `"d"`: Perturb data only
    - `"r"`: Perturb rr only
    - `"dr"`: Perturb both data and rr
- `early_stop`: (Optional) Integer. If set, stops after the specified number of successful attacks.


## References

- [VPNet Paper/Repository](#) (add link if available)

