from . import fgsm
from .params import MethodParams, SetupParams

methods = {
    "FGSM" : fgsm.FGSM(),


}

__all__ = ["methods", "MethodParams", "SetupParams"]
