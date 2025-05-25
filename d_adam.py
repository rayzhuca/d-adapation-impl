import torch
from torch.optim import Optimizer
import torch.distributed as dist

class DAdam(Optimizer):
    
    def __init__(self, params, eps = 1e-8, d0 = 1e-6, lr = 1.0, betas = (0.9, 0.999)):
        defaults = dict(
            eps=eps,
            k=0,
            d=d0,
            lr=lr,
            betas=betas,
            growth_rate=float('inf'),
            numerator_weighted=0.0
        )
        super().__init__(params, defaults)

    def step(self):
        sk_l1 = 0.0

        group = self.param_groups[0]
        numerator_weighted = group['numerator_weighted']
        beta1, beta2 = group['betas']

        k = group['k']
        d = group['d']
        lr = max(group['lr'] for group in self.param_groups)
        dlr = d*lr
        growth_rate = group['growth_rate']
        sqrt_beta2 = beta2**(0.5)
        numerator_acum = 0.0

        for group in self.param_groups:
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data).detach()
                    state['exp_avg'] = torch.zeros_like(p.data).detach()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                s = state['s']

                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += dlr * torch.dot(grad.flatten(), s.div(denom).flatten()).item()

                    exp_avg.mul_(beta1).add_(grad, alpha=dlr*(1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr*(1-sqrt_beta2))
                    sk_l1 += s.abs().sum().item()

        d_hat = d

        if sk_l1 == 0:
            return        
  
        global_numerator_weighted = sqrt_beta2*numerator_weighted + (1-sqrt_beta2)*numerator_acum
        global_sk_l1 = sk_l1

        if lr > 0.0:
            d_hat = global_numerator_weighted/((1-sqrt_beta2)*global_sk_l1)
            d = max(d, min(d_hat, d*growth_rate))

        for group in self.param_groups:
            group['numerator_weighted'] = global_numerator_weighted
            group['d'] = d

            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group['k'] = k + 1