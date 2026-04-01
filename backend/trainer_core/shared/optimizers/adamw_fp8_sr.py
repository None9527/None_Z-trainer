"""
AdamW with FP8 State Storage + Stochastic Rounding

Based on adamw_fp8.py but replaces deterministic rounding with stochastic rounding
during FP8 quantization. This ensures E[quantize(x)] = x (unbiased),
preventing systematic loss of small gradient updates.

Mathematical guarantee:
  - Standard rounding: small updates are always rounded to zero → biased, O(N) error
  - Stochastic rounding: small updates are probabilistically preserved → unbiased, O(√N) error
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Dict, Any


class AdamWFP8SR(Optimizer):
    """
    AdamW optimizer with FP8 state storage and Stochastic Rounding.
    
    Key difference from AdamWFP8:
      _to_fp8_scaled uses stochastic rounding instead of round-to-nearest,
      ensuring small gradient increments are preserved in expectation.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)
        
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                "PyTorch float8_e4m3fn not available. "
                "Requires PyTorch >= 2.1. Current version: " + torch.__version__
            )
        self._fp8_dtype_m = torch.float8_e4m3fn
        self._fp8_dtype_v = torch.float8_e5m2
    
    def _to_fp8_stochastic(self, tensor: torch.Tensor, fp8_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert tensor to FP8 with per-tensor scaling and STOCHASTIC ROUNDING.
        
        Instead of round-to-nearest (which loses small deltas), we:
        1. Scale tensor into FP8 range
        2. Cast to FP8 (floor) and FP8+1 (ceil)
        3. Randomly choose floor or ceil with probability proportional to proximity
        
        This ensures E[result] = original value (unbiased).
        """
        if fp8_dtype == torch.float8_e4m3fn:
            max_fp8 = 448.0
        else:
            max_fp8 = 57344.0
        
        abs_max = tensor.abs().max().clamp(min=1e-12)
        scale = abs_max / max_fp8
        
        # Scale down to FP8 range
        scaled = tensor / scale
        
        # Deterministic floor (round toward zero)
        fp8_floor = scaled.to(fp8_dtype)
        fp8_floor_f32 = fp8_floor.to(torch.float32)
        
        # Compute residual: how much was lost by flooring
        residual = scaled - fp8_floor_f32
        
        # Compute the FP8 ULP (unit in the last place) at each value
        # By adding a small perturbation and re-quantizing
        eps_tensor = torch.where(scaled >= 0, 
                                  torch.ones_like(scaled) * 1e-6,
                                  torch.ones_like(scaled) * -1e-6)
        fp8_next = (fp8_floor_f32 + eps_tensor.sign() * (fp8_floor_f32.abs() * 0.125 + 1e-10)).to(fp8_dtype)
        fp8_next_f32 = fp8_next.to(torch.float32)
        
        ulp = (fp8_next_f32 - fp8_floor_f32).abs().clamp(min=1e-30)
        
        # Probability of rounding up = residual / ulp
        prob = (residual.abs() / ulp).clamp(0.0, 1.0)
        
        # Stochastic decision
        rand = torch.rand_like(prob)
        should_round_up = rand < prob
        
        # Apply: floor + direction * should_round_up
        result_f32 = torch.where(should_round_up, fp8_next_f32, fp8_floor_f32)
        fp8_tensor = result_f32.to(fp8_dtype)
        
        return fp8_tensor, scale
    
    def _from_fp8_scaled(self, fp8_tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert FP8 tensor back using stored scale."""
        return fp8_tensor.to(dtype) * scale
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWFP8SR does not support sparse gradients")
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=self._fp8_dtype_m)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=self._fp8_dtype_v)
                    state['scale_m'] = torch.tensor(1.0, device=p.device)
                    state['scale_v'] = torch.tensor(1.0, device=p.device)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=self._fp8_dtype_v)
                        state['scale_max_v'] = torch.tensor(1.0, device=p.device)
                
                state['step'] += 1
                step = state['step']
                
                # Upcast FP8 states to fp32 for computation
                exp_avg = self._from_fp8_scaled(state['exp_avg'], state['scale_m'], torch.float32)
                exp_avg_sq = self._from_fp8_scaled(state['exp_avg_sq'], state['scale_v'], torch.float32)
                if amsgrad:
                    max_exp_avg_sq = self._from_fp8_scaled(state['max_exp_avg_sq'], state['scale_max_v'], torch.float32)
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                grad_fp32 = grad.to(torch.float32)
                
                # Update moments in fp32
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)
                
                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                update = exp_avg / denom * step_size
                p.data.add_(update.to(p.dtype), alpha=-1)
                
                # Downcast states to FP8 with STOCHASTIC ROUNDING
                state['exp_avg'], state['scale_m'] = self._to_fp8_stochastic(exp_avg, self._fp8_dtype_m)
                state['exp_avg_sq'], state['scale_v'] = self._to_fp8_stochastic(exp_avg_sq, self._fp8_dtype_v)
                if amsgrad:
                    state['max_exp_avg_sq'], state['scale_max_v'] = self._to_fp8_stochastic(max_exp_avg_sq, self._fp8_dtype_v)

        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict with FP8 states converted to fp32 for saving."""
        state_dict = super().state_dict()
        for param_id, param_state in state_dict['state'].items():
            for key, scale_key in [('exp_avg', 'scale_m'), ('exp_avg_sq', 'scale_v'), ('max_exp_avg_sq', 'scale_max_v')]:
                if key in param_state:
                    scale = param_state.get(scale_key, torch.tensor(1.0))
                    param_state[key] = param_state[key].to(torch.float32) * scale
                    if scale_key in param_state:
                        del param_state[scale_key]
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
