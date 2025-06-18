from .AttackBase import AttackBase


class FGSM(AttackBase):
    """
    Fast Gradient Sign Method (FGSM) attack.
    """

    def setup(self, params):
        """
        Setup the FGSM attack with the model and data.
        """
        params.model.zero_grad()
        params.data.requires_grad = True
        params.rr.requires_grad = True

    def get_noise(self, params):
        """
        Generate noise based on the model's gradients.
        """
        data_grad = params.data.grad
        rr_grad = params.rr.grad

        return data_grad.sign(), rr_grad.sign()

    @property
    def name(self) -> str:
        return "FGSM"