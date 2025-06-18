from . import fgsm, pgd, bim, cw, deepfool
from .params import MethodParams, SetupParams

methods = {
    "FGSM": fgsm.FGSM(),
    "PGD": pgd.PGD(num_steps=10, step_size=None, random_start=True),
    "PGD_20": pgd.PGD(num_steps=20, step_size=None, random_start=True),
    "PGD_norand": pgd.PGD(num_steps=10, step_size=None, random_start=False),
    "BIM": bim.BIM(num_steps=10),
    "BIM_20": bim.BIM(num_steps=20),
    "CW": cw.CW(num_steps=100, learning_rate=0.01, c=1.0),
    "CW_strong": cw.CW(num_steps=1000, learning_rate=0.01, c=10.0),
    "DeepFool": deepfool.DeepFool(num_classes=5, max_iter=50),
}

__all__ = ["methods", "MethodParams", "SetupParams"]
