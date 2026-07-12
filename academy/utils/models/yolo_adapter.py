"""Ultralytics YOLO backend: segmentation or classification, implementing
the shared :class:`~utils.models.base.ModelAdapter` interface.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from utils.commons import get_device
from utils.plotter import TrainingPlotter
from utils.stages import StageAction, StageController


class YoloAdapter:
    """Adapter exposing the shared ModelAdapter interface for Ultralytics."""

    def __init__(self, config):
        from ultralytics import YOLO

        self.cfg = config
        self.checkpoint_path = config.model.checkpoint or config.model.yolo.variant + ".pt"
        self.model = YOLO(self.checkpoint_path)
        # Ultralytics derives this from the checkpoint's actual architecture
        # (guess_model_task), not its filename -- unlike a "-seg"/"-cls"
        # suffix heuristic, it's correct even for arbitrarily-named checkpoints
        # (e.g. config.yaml's model.checkpoint override).
        self.task = self.model.task or "detect"

    def export_onnx(self, target: str) -> None:
        exported = self.model.export(
                format="onnx",
                imgsz=self.cfg.inference.imgsz,
                simplify=False,  # onnxsim runs as its own later pipeline stage, not duplicated here
                nms=True,  # bakes NMS into the graph so output0 is (1, 300, 4+1+1+nm) already top-K filtered, matching utils/benchmark/onnx_decode.py's decoder
            )
        shutil.move(str(exported), target)

    def prune(self, checkpoint_path: Path, output_path: Path, amount: float) -> Path:
        """Prune the EMA weights inside an Ultralytics checkpoint.

        ``ckpt["ema"]`` is a live, already-instantiated ``nn.Module`` (Ultralytics
        keeps ``ckpt["model"]`` as ``None`` on saved checkpoints — resume/export
        always derive from EMA), so pruning is just: unwrap it, prune its
        ``state_dict()``, load the pruned weights back in, save the same dict.
        """
        import torch

        from utils.optimizers.pruner import prune_state_dict

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        module = ckpt.get("ema") or ckpt.get("model")
        if module is None:
            raise ValueError(f"{checkpoint_path}: checkpoint has neither 'ema' nor 'model' weights to prune")

        pruned_state, stats = prune_state_dict(module.state_dict(), amount)
        module.load_state_dict(pruned_state)
        print(
            f"pruned {stats.pruned_params}/{stats.total_params} params "
            f"({stats.sparsity:.1%} sparsity) across {stats.eligible_tensors} tensors"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(output_path))
        return output_path

    def detect_task(self) -> str:
        return self.task

    def apply_freeze(self, freeze_mode: str, unfreeze_fraction: float = 0.3) -> int:
        """Translate a freeze mode into an Ultralytics ``freeze=`` layer count.

        Returns:
            The number of leading layers to freeze, passed straight to
            ``model.train(freeze=...)``.
        """
        total_layers = len(list(self.model.model.model))
        if freeze_mode == "none":
            return 0
        if freeze_mode == "backbone":
            # Freeze everything except the final ~4 layers (head + last neck fusion).
            return max(total_layers - 4, 0)
        if freeze_mode == "partial":
            return max(int(total_layers * (1 - unfreeze_fraction)), 0)
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def run_stage(
        self,
        stage,
        plotter: TrainingPlotter,
        stage_controller: StageController | None = None,
    ) -> dict[str, Any]:
        """Train one stage. Non-final stages exit early via `stage_controller`
        (plateau -> ``trainer.stop = True``); the final stage relies on
        Ultralytics' own ``patience=`` early stopping.

        # Ponytail: `trainer.stop = True` is the documented community
        # pattern for callback-driven early exit in Ultralytics' BaseTrainer.
        # It has shifted slightly across releases — if a stage runs past
        # its plateau point, check `trainer.stop` semantics for your
        # installed `ultralytics` version.
        """
        is_final = stage_controller is None
        freeze_count = self.apply_freeze(stage.freeze, stage.unfreeze_fraction)
        loss_key = "val/seg_loss" if self.task == "segment" else "val/loss"

        # Ultralytics accumulates callbacks across repeated model.train()
        # calls on the same instance; clear this event before re-adding so
        # a previous stage's hook doesn't keep firing in this one.
        self.model.callbacks.setdefault("on_fit_epoch_end", []).clear()

        def on_fit_epoch_end(trainer):
            val_loss = trainer.metrics.get(loss_key)
            if val_loss is None:
                return
            plotter.log(trainer.epoch + 1, {f"{stage.name}/val_loss": val_loss})
            if not is_final and stage_controller.step(val_loss) is StageAction.ADVANCE:
                print(
                    f"\n  [Stage:{stage.name}] validation plateaued — advancing early."
                )
                trainer.stop = True

        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        self.model.train(
            data=str(self.cfg.dataset.yaml_path),
            epochs=stage.max_epochs,
            imgsz=self.cfg.model.yolo.imgsz,
            batch=self.cfg.training.batch_size,
            device=get_device(),
            freeze=freeze_count,
            lr0=self.cfg.training.base_lr * stage.lr_factor,
            optimizer=self.cfg.training.optimizer,
            weight_decay=self.cfg.training.weight_decay,
            warmup_epochs=self.cfg.training.warmup_epochs,
            patience=stage.early_stopping_patience if is_final else stage.max_epochs,
            workers=0,
            project=str(
                Path(self.cfg.output_dir)
                / self.cfg.framework
                / self.cfg.model.yolo.variant
            ),
            name=stage.name,
            exist_ok=True,
            **self.cfg.augmentation,
        )
        save_dir = Path(self.model.trainer.save_dir)
        best_ckpt = save_dir / "weights" / "best.pt"

        # Ultralytics does not support calling .train() again on the same
        # YOLO instance once a run has finished -- self.overrides loses its
        # "model" key and the next .train() call raises KeyError('model').
        # Reload a fresh instance from this stage's best checkpoint so the
        # learned weights still carry forward into the next stage.
        if best_ckpt.exists():
            from ultralytics import YOLO

            self.model = YOLO(str(best_ckpt))

        return {"stage": stage.name, "results_dir": str(save_dir)}

    def predict(self, image: np.ndarray) -> Any:
        return self.model.predict(
            image,
            imgsz=self.cfg.inference.imgsz,
            conf=self.cfg.inference.conf_threshold,
            verbose=False,
        )[0]

    def predict_segmentation_masks(
        self, image: np.ndarray, num_classes: int
    ) -> np.ndarray:
        h, w = image.shape[:2]
        result = self.predict(image)
        pred = np.zeros((num_classes, h, w), dtype=np.uint8)
        if result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for inst_mask, cls_id in zip(mask_data, cls_ids):
                if cls_id >= num_classes:
                    continue
                resized = cv2.resize(
                    inst_mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                pred[cls_id] |= (resized > 0.5).astype(np.uint8)
        return pred

    def predict_classification_label(self, image: np.ndarray) -> int:
        return int(self.predict(image).probs.top1)
