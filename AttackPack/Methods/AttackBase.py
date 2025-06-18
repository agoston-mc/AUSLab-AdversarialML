from .params import MethodParams, SetupParams
from abc import ABC, abstractmethod

class AttackBase(ABC):
    """
    Base class for all attack methods.
    """

    @abstractmethod
    def setup(self, params: SetupParams) -> None:
        """
        Setup the attack method with the model and data.
        """
        pass

    @abstractmethod
    def get_noise(self, params: MethodParams) -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Generate noise based on the model's gradients.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the attack method.
        """
        pass