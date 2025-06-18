import torch
from .AttackBase import AttackBase
from .params import MethodParams, SetupParams


class BIM(AttackBase):
    """
    Basic Iterative Method (BIM) attack.
    Also known as Iterative FGSM (I-FGSM).
    """

    def __init__(self, num_steps=10, step_size=None):
        self.num_steps = num_steps
        self.step_size = step_size  # Will be set to epsilon/num_steps if None
        self.original_data = None
        self.original_rr = None

    def setup(self, params: SetupParams) -> None:
        """
        Setup the BIM attack with the model and data.
        """
        params.model.zero_grad()
        params.data.requires_grad = True
        params.rr.requires_grad = True

        # Store original data for clipping
        self.original_data = params.data.clone().detach()
        self.original_rr = params.rr.clone().detach()

    def get_noise(self, params: MethodParams) -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Generate noise using iterative BIM attack.
        """
        epsilon = getattr(params, 'epsilon', 0.01)
        step_size = self.step_size if self.step_size is not None else epsilon / self.num_steps

        # Start with original data
        adv_data = self.original_data.clone()
        adv_rr = self.original_rr.clone()

        for step in range(self.num_steps):
            adv_data.requires_grad = True
            adv_rr.requires_grad = True

            # Forward pass
            output, ln = params.model(adv_data, adv_rr)

            # Calculate loss
            loss = params.criterion((output, ln), params.target_output)

            # Backward pass
            params.model.zero_grad()
            loss.backward()

            # Get gradients
            data_grad = adv_data.grad.detach()
            rr_grad = adv_rr.grad.detach()

            # Update adversarial examples
            with torch.no_grad():
                adv_data = adv_data + step_size * data_grad.sign()
                adv_rr = adv_rr + step_size * rr_grad.sign()

                # Clip to epsilon ball around original data
                adv_data = torch.clamp(adv_data,
                                       self.original_data - epsilon,
                                       self.original_data + epsilon)
                adv_rr = torch.clamp(adv_rr,
                                     self.original_rr - epsilon,
                                     self.original_rr + epsilon)

        # Return the noise
        data_noise = adv_data - self.original_data
        rr_noise = adv_rr - self.original_rr

        return data_noise, rr_noise

    @property
    def name(self) -> str:
        return f"BIM_{self.num_steps}"

