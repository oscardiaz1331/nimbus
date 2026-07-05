"""RF-DETR (DINOv2 backbone) segmentation backend.

Classification is intentionally unsupported here — RF-DETR is a
detection/segmentation library, and :class:`utils.config.Config` already
rejects ``framework='rfdetr'`` + ``task='classification'`` at load time.
The ``model.rfdetr.segmentation`` flag exists per the spec as the
explicit task toggle; this adapter only implements the ``True``
(segmentation) path and fails loudly otherwise rather than guessing.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from utils.plotter import TrainingPlotter
from utils.stages import StageAction, StageController


class RFDETRAdapter:
    """Adapter exposing the shared ModelAdapter interface for RF-DETR-seg."""

    def __init__(self, config):
        if not config.model.rfdetr.segmentation:
            raise NotImplementedError(
                "model.rfdetr.segmentation=false (plain detection) is out of "
                "scope for this pipeline — set it to true, or switch to "
                "framework='yolo' for classification/detection tasks."
            )

        from rfdetr import RFDETRSegMedium, RFDETRSegSmall, RFDETRSegNano
        from rfdetr.config import (
            RFDETRSegMediumConfig,
            RFDETRSegSmallConfig,
            RFDETRSegNanoConfig,
            SegmentationTrainConfig,
        )

        self.cfg = config
        self.checkpoint = config.model.checkpoint
        match config.model.rfdetr.variant:
            case "nano":
                self._SegConfig = RFDETRSegNanoConfig
                self._SegModel = RFDETRSegNano
                if self.checkpoint is None:
                    self.checkpoint = "rf-detr-seg-nano.pt"  # RF-DETR's official pretrained weights for nano variant
            case "small":
                self._SegConfig = RFDETRSegSmallConfig
                self._SegModel = RFDETRSegSmall
                if self.checkpoint is None:
                    self.checkpoint = "rf-detr-seg-small.pt"  # RF-DETR's official pretrained weights for small variant
            case "medium":
                self._SegConfig = RFDETRSegMediumConfig
                self._SegModel = RFDETRSegMedium
                if self.checkpoint is None:
                    self.checkpoint = "rf-detr-seg-medium.pt"  # RF-DETR's official pretrained weights for medium variant
            case _:
                raise ValueError(
                    f"Unsupported RF-DETR variant: {config.model.rfdetr.variant}"
                )
        self._TrainConfig = SegmentationTrainConfig
        self.model = None  # lazy init on first inference call

    def export_onnx(self, target: str) -> None:
        if self.model is None:
            self.model = self._SegModel(
                pretrain_weights=self.checkpoint,
                resolution=self._SegConfig().resolution,
            )
        self.model.optimize_for_inference()
        output_dir = str(
        Path(self.cfg.output_dir)
        / self.cfg.framework
        / self.cfg.model.rfdetr.variant)
        self.model.export(format="onnx", output_dir=output_dir)
        task = self.detect_task()
        if task == "segment":
            task_str = "seg"
        else:
            print(f"Warning: unexpected task '{task}' during ONNX export; defaulting to 'seg'")
            task_str = "seg"

        shutil.move(str(Path(output_dir) / f"rfdetr-{task_str}-{self.cfg.model.rfdetr.variant}.onnx"), target)

    def detect_task(self) -> str:
        return "segment"

    def apply_freeze(
        self,
        module,
        freeze_mode: str,
        unfreeze_fraction: float = 0.3,
    ) -> None:
        """
        Freeze the pretrained DINOv2 encoder according to the training stage.

        Stages
        ------
        backbone
            Freeze the whole pretrained DINOv2 encoder.
            Train the RF-DETR projector, transformer and prediction heads.

        partial
            Freeze the early DINOv2 blocks and progressively unfreeze the last
            Transformer blocks. The projector and all RF-DETR modules remain
            trainable.

        none
            Train the whole network.
        """

        params = list(module.model.named_parameters())

        # Start from a fully trainable model
        for _, param in params:
            param.requires_grad = True

        if freeze_mode == "none":
            return

        encoder_prefix = "backbone.0.encoder"

        # ------------------------------------------------------------------
        # Stage 1
        # Freeze the pretrained DINOv2 encoder.
        # ------------------------------------------------------------------

        if freeze_mode == "backbone":

            for name, param in params:
                if name.startswith(encoder_prefix):
                    param.requires_grad = False

            return

        # ------------------------------------------------------------------
        # Stage 2
        # Freeze the first Transformer blocks and unfreeze the last ones.
        # ------------------------------------------------------------------

        if freeze_mode != "partial":
            raise ValueError(f"Unknown freeze mode: {freeze_mode}")

        layer_regex = re.compile(
            r"backbone\.0\.encoder\.encoder\.encoder\.layer\.(\d+)"
        )

        layer_ids = sorted(
            {
                int(match.group(1))
                for name, _ in params
                if (match := layer_regex.search(name))
            }
        )

        if not layer_ids:
            raise RuntimeError("Could not locate DINOv2 Transformer layers.")

        total_layers = len(layer_ids)

        num_trainable = max(1, round(total_layers * unfreeze_fraction))
        first_trainable = total_layers - num_trainable

        # Freeze the complete DINOv2 encoder first
        for name, param in params:
            if name.startswith(encoder_prefix):
                param.requires_grad = False

        # Unfreeze the last Transformer blocks
        for name, param in params:

            match = layer_regex.search(name)

            if match is None:
                continue

            layer_idx = int(match.group(1))

            if layer_idx >= first_trainable:
                param.requires_grad = True

        # Keep LayerNorm trainable.
        # This usually stabilizes fine-tuning.
        for name, param in params:
            if name.endswith("layernorm.weight") or name.endswith("layernorm.bias"):
                param.requires_grad = True

    def run_stage(
        self,
        stage,
        plotter: TrainingPlotter,
        stage_controller: StageController | None = None,
    ) -> dict[str, Any]:
        """Train one stage via a fresh PyTorch Lightning ``Trainer``.

        Non-final stages exit early when validation plateaus, via the
        official Lightning API ``trainer.should_stop = True`` set from a
        callback. The final stage relies on RF-DETR's native
        ``early_stopping`` / ``early_stopping_patience``.
        """
        from pytorch_lightning.callbacks import Callback
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

        is_final = stage_controller is None
        model_config = self._SegConfig(pretrain_weights=self.checkpoint, num_classes=1)
        output_dir = str(
            Path(self.cfg.output_dir)
            / self.cfg.framework
            / self.cfg.model.rfdetr.variant
            / stage.name
        )
        train_config = self._TrainConfig(
            dataset_dir=str(self.cfg.dataset.root),
            output_dir=output_dir,
            epochs=stage.max_epochs,
            batch_size=self.cfg.training.batch_size,
            grad_accum_steps=self.cfg.training.grad_accum,
            lr=self.cfg.training.base_lr * stage.lr_factor,
            lr_encoder=self.cfg.training.base_lr * stage.lr_factor * 0.1,
            resolution=self._SegConfig().resolution,
            early_stopping=is_final,
            early_stopping_patience=stage.early_stopping_patience,
            aug_config=self.cfg.augmentation or None,
        )

        module = RFDETRModelModule(model_config, train_config)
        datamodule = RFDETRDataModule(model_config, train_config)
        trainer = build_trainer(train_config, model_config)

        total = 0
        trainable = 0

        for n, p in module.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        print("Total parameters before freezing:", total)
        print("Trainable parameters before freezing:", trainable)

        self.apply_freeze(module, stage.freeze, stage.unfreeze_fraction)

        total = 0
        trainable = 0

        for n, p in module.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        print("Total parameters after freezing:", total)
        print("Trainable parameters after freezing:", trainable)

        class _StageHooks(Callback):
            def on_validation_epoch_end(self, trainer, pl_module):
                val_loss = trainer.callback_metrics.get("val/loss")
                if val_loss is None:
                    return
                plotter.log(
                    trainer.current_epoch + 1,
                    {f"{stage.name}/val_loss": float(val_loss)},
                )
                if (
                    not is_final
                    and stage_controller.step(float(val_loss)) is StageAction.ADVANCE
                ):
                    print(
                        f"\n  [Stage:{stage.name}] validation plateaued — advancing early."
                    )
                    trainer.should_stop = True

        trainer.callbacks.append(_StageHooks())
        trainer.fit(module, datamodule)
        self.checkpoint = str(Path(output_dir) / "checkpoint_best_ema.pth")
        return {"stage": stage.name, "output_dir": train_config.output_dir}

    def predict(self, image: np.ndarray) -> Any:
        if self.model is None:
            self.model = self._SegModel(
                pretrain_weights=self.checkpoint,
                resolution=self._SegConfig().resolution,
            )
            self.model.optimize_for_inference()
        return self.model.predict(image, threshold=self.cfg.inference.conf_threshold)

    def predict_segmentation_masks(
        self, image: np.ndarray, num_classes: int
    ) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.predict(image_rgb)
        pred = np.zeros((num_classes, h, w), dtype=np.uint8)
        if detections.mask is not None:
            for inst_mask, cls_id in zip(detections.mask, detections.class_id):
                cls_id = int(cls_id)
                if cls_id >= num_classes:
                    continue
                if inst_mask.shape != (h, w):
                    inst_mask = cv2.resize(
                        inst_mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                pred[cls_id] |= inst_mask.astype(np.uint8)
        return pred

    def predict_classification_label(self, image: np.ndarray) -> int:
        raise NotImplementedError(
            "RF-DETR backend is segmentation-only in this pipeline."
        )
