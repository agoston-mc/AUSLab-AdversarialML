import torch
from .AttackBase import AttackBase
from .params import MethodParams, SetupParams


class PGD(AttackBase):
    """
    Projected Gradient Descent (PGD) attack.
    Multi-step iterative attack with projection to epsilon ball.
    """

    def __init__(self, num_steps=10, step_size=None, random_start=True):
        self.num_steps = num_steps
        self.step_size = step_size  # Will be set to epsilon/4 if None
        self.random_start = random_start
        self.original_data = None
        self.original_rr = None

    def setup(self, params: SetupParams) -> None:
        """
        Setup the PGD attack with the model and data.
        """
        params.model.zero_grad()
        params.data.requires_grad = True
        params.rr.requires_grad = True

        # Store original data for projection
        self.original_data = params.data.clone().detach()
        self.original_rr = params.rr.clone().detach()

    def get_noise(self, params: MethodParams) -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Generate noise using iterative PGD attack.
        """
        epsilon = getattr(params, 'epsilon', 0.01)  # Should be passed in params
        step_size = self.step_size if self.step_size is not None else epsilon / 4

        # Initialize adversarial examples
        if self.random_start:
            # Start with random noise in [-epsilon, epsilon]
            data_noise = torch.empty_like(params.data).uniform_(-epsilon, epsilon)
            rr_noise = torch.empty_like(params.rr).uniform_(-epsilon, epsilon)
        else:
            data_noise = torch.zeros_like(params.data)
            rr_noise = torch.zeros_like(params.rr)

        adv_data = self.original_data + data_noise
        adv_rr = self.original_rr + rr_noise

        for step in range(self.num_steps):
            adv_data.requires_grad = True
            adv_rr.requires_grad = True

            # Forward pass
            output, ln = params.model(adv_data, adv_rr)

            # Calculate loss (we want to maximize it for untargeted attack)
            loss = params.criterion((output, ln), params.target_output)

            # Backward pass
            params.model.zero_grad()
            loss.backward()

            # Get gradients
            data_grad = adv_data.grad.detach()
            rr_grad = adv_rr.grad.detach()

            # Update adversarial examples
            with torch.no_grad():
                # Take step in direction of gradient (to maximize loss)
                adv_data = adv_data + step_size * data_grad.sign()
                adv_rr = adv_rr + step_size * rr_grad.sign()

                # Project back to epsilon ball
                data_noise = adv_data - self.original_data
                rr_noise = adv_rr - self.original_rr

                data_noise = torch.clamp(data_noise, -epsilon, epsilon)
                rr_noise = torch.clamp(rr_noise, -epsilon, epsilon)

                adv_data = self.original_data + data_noise
                adv_rr = self.original_rr + rr_noise

        return data_noise, rr_noise

    @property
    def name(self) -> str:
        return f"PGD_{self.num_steps}"

