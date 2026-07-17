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

    def prune(self, checkpoint_path: Path, output_path: Path, amount: float) -> Path:
        """Prune the weights inside an RF-DETR checkpoint.

        Unlike YOLO's checkpoint, ``ckpt["model"]`` here is already a flat
        ``state_dict`` (raw tensors, no live module) — both the pretrained
        downloads and the PTL trainer's own saves (see
        ``BestModelCallback._build_checkpoint_payload``) use this shape. When a
        ``"state_dict"`` key is also present (the PTL-resumable view, same
        tensors under ``"model."``-prefixed keys) it's rebuilt from the pruned
        weights so the two views can't drift apart.
        """
        import torch

        from utils.optimizers.pruner import prune_state_dict

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if "model" not in ckpt:
            raise ValueError(f"{checkpoint_path}: checkpoint has no 'model' key to prune")

        pruned_state, stats = prune_state_dict(ckpt["model"], amount)
        ckpt["model"] = pruned_state
        if "state_dict" in ckpt:
            ckpt["state_dict"] = {f"model.{k}": v for k, v in pruned_state.items()}
        print(
            f"pruned {stats.pruned_params}/{stats.total_params} params "
            f"({stats.sparsity:.1%} sparsity) across {stats.eligible_tensors} tensors"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(output_path))
        return output_path

    def export_onnx(self, target: str) -> None:
        if self.model is None:
            self.model = self._SegModel(
                pretrain_weights=self.checkpoint,
                resolution=self._SegConfig().resolution,
                num_classes=1,
            )
        if not self.cfg.model.rfdetr.export_fallback:
            print("Exporting RF-DETR model to ONNX using rfdetr library")
            try:
                self._export_native(target)
            except BaseException as e:
                print(type(e))
                print(f"Error occurred while exporting ONNX model: {e}")
                print(
                    "To export the model to ONNX, you can try the hand-rolled wrapper instead by "
                    "setting model.rfdetr.export_fallback: true in config.yaml."
                )
                raise
        else:
            print("Exporting RF-DETR model to ONNX with FrancescoCappio's wrapper")
            self._export_via_wrapper(target)
        print(f"Model exported with post-processing to {target}")

    def _export_native(self, target: str) -> None:
        self.model.optimize_for_inference()
        onnx_path = self.model.export()
        shutil.move(onnx_path, target)

    def _export_via_wrapper(self, target: str) -> None:
        """Hand-rolled ONNX export, bypassing rfdetr's own ``.export()``.

        FrancescoCappio: https://github.com/roboflow/rf-detr/issues/376#issuecomment-3852504941
        Manual escape hatch for when the rfdetr library's native export breaks —
        reimplements the forward pass (sigmoid + mask upsample) and applies a
        LayerNorm patch the linked issue found necessary for dynamic-batch ONNX
        export, working against whichever variant/checkpoint this adapter was
        configured with rather than a hardcoded model class.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from rfdetr.models.backbone import projector

        # A LayerNorm fix is needed for dynamic batching support.
        # We apply the fix before creating the model.
        def fixed_forward(self, x):
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

        projector.LayerNorm.forward = fixed_forward

        class RFDETRSegONNXWrapper(nn.Module):
            def __init__(self, rfdetr_model):
                super().__init__()
                self.model = rfdetr_model.model.model  # Access the underlying PyTorch module
                self.resolution = rfdetr_model.model.resolution

            def forward(self, x):
                # Standard forward pass. Returns a tuple, not a dict, in export mode:
                # (pred_boxes, pred_logits, pred_masks).
                outputs = self.model(x)

                boxes = outputs[0]
                logits = outputs[1]
                mask_logits = outputs[2]  # low-res prototypes

                # Post-processing: sigmoid to get probabilities.
                probs = logits.sigmoid()

                # Mask decoding mirrors rfdetr's PostProcess._postprocess_masks:
                # bilinear-upsample the LOGITS (fixed size for ONNX compatibility),
                # sigmoid only afterwards. Upsampling after sigmoid interpolates in
                # saturated probability space, which drags the 0.5 boundary outward
                # and inflates every mask relative to the pytorch path.
                masks = nn.functional.interpolate(
                    mask_logits,
                    size=(self.resolution, self.resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                masks = masks.sigmoid()

                return probs, boxes, masks

        self.model.model.model.eval()  # needed to get the predicted masks from the inner model
        self.model.model.model.export()  # convert layers to their export-able version

        wrapper = RFDETRSegONNXWrapper(self.model)
        wrapper.eval()

        dummy_input = torch.randn(1, 3, self.model.model.resolution, self.model.model.resolution)

        torch.onnx.export(
            wrapper,
            dummy_input,
            target,
            export_params=True,
            opset_version=17,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["scores", "boxes", "masks"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "scores": {0: "batch_size"},
                "boxes": {0: "batch_size"},
                "masks": {0: "batch_size"},
            },
            verbose=False,
        )
