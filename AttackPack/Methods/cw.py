import torch
from .AttackBase import AttackBase
from .params import MethodParams, SetupParams


class CW(AttackBase):
    """
    Carlini & Wagner (C&W) L2 attack.
    Uses optimization-based approach with tanh transformation.
    """

    def __init__(self, num_steps=1000, learning_rate=0.01, c=1.0, kappa=0):
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.c = c  # Regularization parameter
        self.kappa = kappa  # Confidence parameter
        self.original_data = None
        self.original_rr = None

    def setup(self, params: SetupParams) -> None:
        """
        Setup the C&W attack with the model and data.
        """
        # Store original data
        self.original_data = params.data.clone().detach()
        self.original_rr = params.rr.clone().detach()

    def get_noise(self, params: MethodParams) -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Generate noise using C&W optimization attack.
        """
        # Initialize perturbation variables in tanh space
        w_data = torch.zeros_like(self.original_data, requires_grad=True)
        w_rr = torch.zeros_like(self.original_rr, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([w_data, w_rr], lr=self.learning_rate)

        # Get target class (for untargeted attack, we want to maximize loss)
        target_class = params.target_output.argmax(dim=-1)

        best_data_noise = torch.zeros_like(self.original_data)
        best_rr_noise = torch.zeros_like(self.original_rr)
        best_distance = float('inf')

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Transform from tanh space to input space
            # This ensures perturbations stay in valid range
            adv_data = self.original_data + 0.5 * (torch.tanh(w_data) + 1) - self.original_data
            adv_rr = self.original_rr + 0.5 * (torch.tanh(w_rr) + 1) - self.original_rr

            # Forward pass
            output, ln = params.model(adv_data, adv_rr)

            # C&W loss function
            # L2 distance term
            l2_distance = torch.norm(adv_data - self.original_data) + torch.norm(adv_rr - self.original_rr)

            # Classification loss term
            logits = output
            real_class_logit = logits[0, target_class]

            # Find the maximum logit for other classes
            other_logits = logits.clone()
            other_logits[0, target_class] = -float('inf')
            max_other_logit = torch.max(other_logits)

            # f(x) function from C&W paper (for untargeted attack)
            f_loss = torch.clamp(real_class_logit - max_other_logit + self.kappa, min=0)

            # Total loss
            loss = l2_distance + self.c * f_loss

            loss.backward()
            optimizer.step()

            # Check if this is the best adversarial example so far
            with torch.no_grad():
                current_distance = l2_distance.item()
                predicted_class = output.argmax(dim=-1).item()

                if predicted_class != target_class.item() and current_distance < best_distance:
                    best_distance = current_distance
                    best_data_noise = adv_data - self.original_data
                    best_rr_noise = adv_rr - self.original_rr

        return best_data_noise, best_rr_noise

    @property
    def name(self) -> str:
        return f"CW_c{self.c}"
