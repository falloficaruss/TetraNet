import torch

from quaternary import QBitLinearQuaternary


def multi_well_potential(
    model: torch.nn.Module,
    wells: list[float] | None = None,
) -> torch.Tensor:
    if wells is None:
        wells = [0.25, 0.5]

    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for _name, module in model.named_modules():
        if isinstance(module, QBitLinearQuaternary):
            distances = torch.stack([(module.c - w).abs() for w in wells])
            closest_idx = torch.argmin(distances).detach()
            closest_well = torch.as_tensor(wells[closest_idx], device=module.c.device)
            total = total + (module.c - closest_well).abs()
    return total


class AdaptiveSnappingScheduler:
    def __init__(self, alpha: float = 0.02, snap_start: float = 0.9):
        self.alpha = alpha
        self.snap_start = snap_start

    def get_lambda(self, progress: float, task_loss: torch.Tensor) -> float:
        progress = max(progress, 0.0)
        if progress <= 0.0:
            return 0.0
        return self.alpha * task_loss.item() * progress


def compute_total_loss(
    task_loss: torch.Tensor,
    model: torch.nn.Module,
    progress: float,
    scheduler: AdaptiveSnappingScheduler | None = None,
    wells: list[float] | None = None,
) -> torch.Tensor:
    if scheduler is None:
        scheduler = AdaptiveSnappingScheduler()
    lam = scheduler.get_lambda(progress, task_loss.detach())
    penalty = multi_well_potential(model, wells=wells)
    return task_loss + lam * penalty
