import torch

from quaternary import LearnedCQuaternaryLinear


def multi_well_potential(
    model: torch.nn.Module,
    wells: list[float] | None = None,
) -> torch.Tensor:
    """L1 multi-well potential: sum of |c - nearest_well| across all learnable-c layers.

    L1 is used over L2 based on empirical findings:
    L2 gradient decays to zero as c approaches a well, allowing competing
    quantization gradients to halt convergence prematurely.
    L1 maintains a constant gradient of ±1 regardless of distance.
    """
    if wells is None:
        wells = [0.25, 0.5]

    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for _name, module in model.named_modules():
        if isinstance(module, LearnedCQuaternaryLinear):
            distances = torch.stack([(module.c - w).abs() for w in wells])
            closest_idx = torch.argmin(distances).detach()
            closest_well = torch.as_tensor(
                wells[closest_idx], device=module.c.device
            )
            total = total + (module.c - closest_well).abs()
    return total


class AdaptiveSnappingScheduler:
    """Schedules lambda(t) for the snapping penalty.

    Lambda scales linearly with task_loss and progress, starting from 0
    and ramping to alpha * task_loss by snap_start.
    """

    def __init__(self, alpha: float = 2.0, snap_start: float = 0.4):
        self.alpha = alpha
        self.snap_start = snap_start

    def get_lambda(self, progress: float, task_loss: torch.Tensor) -> float:
        if progress < self.snap_start:
            return 0.0
        return self.alpha * task_loss.item() * progress


def compute_total_loss(
    task_loss: torch.Tensor,
    model: torch.nn.Module,
    progress: float,
    scheduler: AdaptiveSnappingScheduler | None = None,
    wells: list[float] | None = None,
) -> torch.Tensor:
    """total_loss = task_loss + lambda(t) * multi_well_potential"""
    if scheduler is None:
        scheduler = AdaptiveSnappingScheduler()
    lam = scheduler.get_lambda(progress, task_loss.detach())
    penalty = multi_well_potential(model, wells=wells)
    return task_loss + lam * penalty
