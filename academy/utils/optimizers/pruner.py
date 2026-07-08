"""Framework-agnostic magnitude pruning over a flat ``{name: tensor}`` mapping.

Operates on plain state-dict-shaped data, not live ``nn.Module`` instances —
YOLO and RF-DETR checkpoints disagree on how weights are stored (see the two
adapters' ``prune()`` methods for the framework-specific unwrap/rewrap), but
once you have a flat tensor mapping, "zero out the smallest-magnitude weights
globally" is identical for both. Global unstructured L1 pruning, not
``torch.nn.utils.prune``: that API reparametrizes live modules with masks,
which buys nothing here since both adapters already hand us plain tensors and
want a plain tensor mapping back.
"""
from __future__ import annotations

import dataclasses

_SKIP_SUBSTRINGS = ("bn.", "norm", "bias", "embed")


@dataclasses.dataclass
class PruneStats:
    total_params: int
    pruned_params: int
    eligible_tensors: int

    @property
    def sparsity(self) -> float:
        return self.pruned_params / self.total_params if self.total_params else 0.0


def _is_prunable(name: str, tensor) -> bool:
    """Conv/Linear-style weights only: >=2D, and not norm/bias/embedding params.

    Pruning a BatchNorm/LayerNorm affine parameter or a bias vector barely
    touches parameter count but reliably wrecks accuracy — those are excluded
    by name rather than by op type since we never see the surrounding graph.
    """
    if tensor.ndim < 2:
        return False
    lname = name.lower()
    return not any(s in lname for s in _SKIP_SUBSTRINGS)


def prune_state_dict(
    state_dict: dict[str, "torch.Tensor"],
    amount: float,
    skip_substrings: tuple[str, ...] = _SKIP_SUBSTRINGS,
) -> tuple[dict[str, "torch.Tensor"], PruneStats]:
    """Zero out the ``amount`` fraction of smallest-magnitude weights, globally.

    "Globally" means the magnitude threshold is computed once across every
    eligible tensor concatenated together, not per-tensor — a layer that's
    already sparser than average gets pruned less than one that isn't,
    instead of every layer losing the same flat fraction.

    Args:
        state_dict: Flat ``{param_name: tensor}`` mapping, e.g. straight off
            ``nn.Module.state_dict()`` or an RF-DETR checkpoint's ``"model"`` key.
        amount: Fraction in ``[0, 1)`` of eligible weights to zero out.
        skip_substrings: Lowercase substrings that exclude a tensor by name
            (norm/bias/embedding params, which shouldn't be pruned).

    Returns:
        A new dict (input is not mutated) with the same keys/shapes, plus
        stats describing what was actually pruned.
    """
    import torch

    if not 0.0 <= amount < 1.0:
        raise ValueError(f"amount must be in [0, 1), got {amount}")

    eligible = {
        name: tensor
        for name, tensor in state_dict.items()
        if _is_prunable(name, tensor) and not any(s in name.lower() for s in skip_substrings)
    }

    pruned = dict(state_dict)
    total_params = sum(t.numel() for t in state_dict.values())

    if not eligible or amount == 0.0:
        return pruned, PruneStats(total_params=total_params, pruned_params=0, eligible_tensors=len(eligible))

    # kthvalue, not quantile: quantile's CUDA/CPU kernel refuses inputs over
    # 2**24 elements, which a DINOv2-backboned RF-DETR blows past easily.
    all_magnitudes = torch.cat([t.detach().abs().flatten() for t in eligible.values()])
    k = max(1, round(amount * all_magnitudes.numel()))
    threshold = torch.kthvalue(all_magnitudes.float(), k).values

    pruned_params = 0
    for name, tensor in eligible.items():
        mask = tensor.detach().abs() > threshold
        pruned[name] = tensor * mask
        pruned_params += int((~mask).sum().item())

    stats = PruneStats(total_params=total_params, pruned_params=pruned_params, eligible_tensors=len(eligible))
    return pruned, stats
