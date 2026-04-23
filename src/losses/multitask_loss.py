import torch
import torch.nn as nn


class MultitaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018).

    Each task i contributes:  (1 / 2σᵢ²) · Lᵢ  +  log σᵢ
    where log σᵢ² is a learnable parameter.
    This automatically balances task contributions during training without
    hand-tuning loss weights.
    """

    def __init__(self, task_names, loss_fns):
        super().__init__()
        self.task_names = task_names
        self.loss_fns   = nn.ModuleDict(loss_fns)
        self.log_vars   = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(self, outputs, targets):
        total  = torch.tensor(0.0, requires_grad=True)
        losses = {}

        for name in self.task_names:
            if name not in outputs or name not in targets:
                continue
            task_loss  = self.loss_fns[name](outputs[name], targets[name])
            precision  = (-self.log_vars[name]).exp()
            weighted   = precision * task_loss + self.log_vars[name]
            total      = total + weighted
            losses[name] = task_loss.detach().item()

        losses['total'] = total
        return total, losses
