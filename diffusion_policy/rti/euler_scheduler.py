from diffusers import DDIMScheduler
from diffusers.utils import BaseOutput
import torch
from typing import List, Optional, Tuple, Union
import numpy as np

from rti.eunms import Epsilon_Update_Type


class RTISchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
 
    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class RTIScheduler(DDIMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[RTISchedulerOutput, Tuple]:
        """
           Newton-Raphson inversion to predict the sample at the previous timestep by reversing the SDE.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model, interpreted as the gradient.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`): current instance of the sample being created by the diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped predicted
                original sample. Necessary because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`, optional): noise for the variance instead of generating it.
            return_dict (`bool`): whether to return `DDIMSchedulerOutput` or a tuple.

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
                A structured output if `return_dict` is `True`, or a tuple otherwise.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None'. You need to run 'set_timesteps' after creating the scheduler."
            )

        # Get the previous timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # Compute alphas and betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample (x_0)
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
            model_output = alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample
        else:
            raise ValueError(
                f"prediction_type {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`."
            )

        # Optionally clip the predicted x_0
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        # Compute the Hessian as the gradient of the model output
        def model_gradient_fn(x):
            x.requires_grad_(True)
            return torch.autograd.grad(model_output.sum(), x, create_graph=True)[0]

        hessian = model_gradient_fn(sample)

        # Newton-Raphson update: x_t-1 = x_t - H^(-1) * grad
        try:
            hessian_inv = torch.linalg.inv(hessian)  # Invert Hessian
            delta_sample = torch.matmul(hessian_inv, model_output.unsqueeze(-1)).squeeze(-1)
            updated_sample = sample - delta_sample
        except torch.linalg.LinAlgError:
            raise RuntimeError("Hessian matrix is singular or ill-conditioned.")

        # # Add noise if eta > 0
        # if eta > 0:
        #     device = model_output.device
        #     if variance_noise is not None and generator is not None:
        #         raise ValueError("Cannot use both `generator` and `variance_noise` simultaneously.")

        #     if variance_noise is None:
        #         if device.type == "mps":
        #             variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
        #             variance_noise = variance_noise.to(device)
        #         else:
        #             variance_noise = torch.randn(
        #                 model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        #             )
        #     variance = self._get_variance(timestep, prev_timestep) ** 0.5 * eta * variance_noise
        #     updated_sample = updated_sample + variance

        if not return_dict:
            return (updated_sample,)

        return RTISchedulerOutput(prev_sample=updated_sample, pred_original_sample=pred_original_sample)
        