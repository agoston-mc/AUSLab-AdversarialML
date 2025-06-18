import torch
from .AttackBase import AttackBase
from .params import MethodParams, SetupParams


class DeepFool(AttackBase):
    """
    DeepFool attack implementation.
    Finds minimal perturbation to cross decision boundary.
    """

    def __init__(self, num_classes=5, overshoot=0.02, max_iter=50):
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.original_data = None
        self.original_rr = None

    def setup(self, params: SetupParams) -> None:
        """
        Setup the DeepFool attack.
        """
        self.original_data = params.data.clone().detach()
        self.original_rr = params.rr.clone().detach()

    def get_noise(self, params: MethodParams) -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Generate minimal noise using DeepFool algorithm.
        """
        adv_data = self.original_data.clone().detach()
        adv_rr = self.original_rr.clone().detach()

        # Get initial prediction
        adv_data.requires_grad = True
        adv_rr.requires_grad = True

        output, ln = params.model(adv_data, adv_rr)
        initial_pred = output.argmax(dim=-1).item()

        iteration = 0
        current_pred = initial_pred

        while current_pred == initial_pred and iteration < self.max_iter:
            # Clear gradients
            if adv_data.grad is not None:
                adv_data.grad.zero_()
            if adv_rr.grad is not None:
                adv_rr.grad.zero_()

            # Forward pass
            output, ln = params.model(adv_data, adv_rr)

            # Calculate gradients for all classes
            gradients_data = []
            gradients_rr = []

            for k in range(self.num_classes):
                if k == initial_pred:
                    continue

                # Clear gradients
                params.model.zero_grad()
                if adv_data.grad is not None:
                    adv_data.grad.zero_()
                if adv_rr.grad is not None:
                    adv_rr.grad.zero_()

                # Compute gradient of (f_k - f_initial_pred)
                score_diff = output[0, k] - output[0, initial_pred]
                score_diff.backward(retain_graph=True)

                gradients_data.append(adv_data.grad.clone())
                gradients_rr.append(adv_rr.grad.clone())

            # Find the closest decision boundary
            distances = []
            for i, (grad_data, grad_rr) in enumerate(zip(gradients_data, gradients_rr)):
                k = i if i < initial_pred else i + 1  # Adjust for skipped initial_pred

                # Calculate w (gradient) and b (bias term)
                w_data = grad_data.flatten()
                w_rr = grad_rr.flatten()
                w = torch.cat([w_data, w_rr])

                f_k = output[0, k].item()
                f_initial = output[0, initial_pred].item()

                # Distance to hyperplane
                distance = abs(f_k - f_initial) / torch.norm(w)
                distances.append((distance, grad_data, grad_rr, k))

            # Find minimum distance
            min_distance, best_grad_data, best_grad_rr, target_class = min(distances, key=lambda x: x[0])

            # Calculate perturbation
            w_data = best_grad_data.flatten()
            w_rr = best_grad_rr.flatten()
            w_norm_sq = torch.norm(w_data) ** 2 + torch.norm(w_rr) ** 2

            f_target = output[0, target_class].item()
            f_initial = output[0, initial_pred].item()

            # Perturbation magnitude
            r_magnitude = (f_target - f_initial) / w_norm_sq

            # Apply perturbation with overshoot
            with torch.no_grad():
                perturbation_data = (1 + self.overshoot) * r_magnitude * best_grad_data
                perturbation_rr = (1 + self.overshoot) * r_magnitude * best_grad_rr

                adv_data = adv_data + perturbation_data
                adv_rr = adv_rr + perturbation_rr

            # Update for next iteration
            adv_data.requires_grad = True
            adv_rr.requires_grad = True

            # Check new prediction
            with torch.no_grad():
                output, _ = params.model(adv_data, adv_rr)
                current_pred = output.argmax(dim=-1).item()

            iteration += 1

        # Return final noise
        data_noise = adv_data - self.original_data
        rr_noise = adv_rr - self.original_rr

        return data_noise.detach(), rr_noise.detach()

    @property
    def name(self) -> str:
        return f"DeepFool_{self.max_iter}"
