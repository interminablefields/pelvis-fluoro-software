from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from PIL import Image
import numpy as np
from perphix.data import PerphixBase

from .pylogger import get_pylogger
from .tensor_utils import batchify

from .dice import DiceLoss2D
from .heatmap_loss import HeatmapLoss2D
from .swin_transformer import SwinTransformerUnet

log = get_pylogger(__name__)


class SwinTransformerUnetModule(LightningModule):
    """LighningModule for training a SWIN module.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int,
        swin_unet: SwinTransformerUnet,
        optimizer: list[torch.optim.Optimizer],
        scheduler: Optional[list[torch.optim.lr_scheduler.LRScheduler]] = None,
    ):
        """
        Args:
            optimizers: The VQ optimizer and the discriminator optimizer, partially initialized
            schedulers: list of schedulers for each optimizer, or None.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        self.swin_unet = swin_unet
        self.dice_loss = DiceLoss2D(skip_bg=False)
        self.heatmap_loss = HeatmapLoss2D()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        opt = self.hparams.optimizer
        opt = opt(params=self.parameters())

        if self.hparams.scheduler is not None:
            sched = self.hparams.scheduler
            sched = sched(optimizer=opt)
            return opt, sched

        return opt

    def forward(self, x: torch.Tensor):
        # TODO: implement forward pass
        outputs = self.swin_unet(x)
        return dict(segs=outputs[:, : self.num_classes], heatmaps=outputs[:, self.num_classes :])

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self,
        batch: Any,
        batch_idx: int,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs: dict[str, torch.Tensor]
        targets: dict[str, torch.Tensor]
        inputs, targets = batch

        imgs = inputs["image"]  # intensity augmentations applied
        target_imgs = targets["image"]  # no intensity augmentations

        outputs = self(imgs)
        decoded_segs = outputs["segs"]
        decoded_heatmaps = outputs["heatmaps"]

        seg_loss = self.dice_loss(decoded_segs, targets["segs"])
        self.log(f"{mode}/seg_loss", seg_loss, on_step=True)

        heatmap_loss = self.heatmap_loss(decoded_heatmaps, targets["heatmaps"])
        self.log(f"{mode}/heatmap_loss", heatmap_loss, on_step=True)

        loss = seg_loss + heatmap_loss

        return loss, decoded_segs, decoded_heatmaps, target_imgs

    def predict(
        self, x: np.ndarray, flip: bool = False
    ) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        """Predict the segmentation and heatmap for a batch of images.

        Note that this method flips the images and outputs, because the model was trained on flipped horizontal images.

        Args:
            x: Either an array of images or a single image, as a float32 in [0,1].

        Returns:
            A dict mapping label names to logits, or a list of such arrays if multiple images provided.

        """
        self.eval()
        x = (x - 0.5) / 0.5  # normalize to [-1, 1]
        x = batchify(x)
        x = np.transpose(x, (0, 3, 1, 2))
        if flip:
            # Flip the images horizontally
            x = np.flip(x, axis=3).copy()
        x = torch.from_numpy(x).float()
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(self.swin_unet.img_size, self.swin_unet.img_size))
        x = x.to(self.device)
        outputs = self(x)

        # Flip the outputs horizontally
        if flip:
            outputs["segs"] = torch.flip(outputs["segs"], dims=[3])
            outputs["heatmaps"] = torch.flip(outputs["heatmaps"], dims=[3])

        # Resize to original image size
        outputs["segs"] = F.interpolate(outputs["segs"], size=(h, w))
        
        outputs["heatmaps"] = F.interpolate(outputs["heatmaps"], size=(h, w))

        outs = []
        for n in range(x.shape[0]):
            decoded_segs = outputs["segs"][n].detach().cpu().numpy()
            decoded_heatmaps = outputs["heatmaps"][n].detach().cpu().numpy()
            out = dict()
            for c in range(decoded_segs.shape[0]):
                name = PerphixBase.get_annotation_name_from_label(c)
                out[f"seg_{name}"] = decoded_segs[c]
            for k in range(decoded_heatmaps.shape[0]):
                name = PerphixBase.get_keypoint_name(k)
                out[f"heatmap_{name}"] = decoded_heatmaps[k]
            outs.append(out)

        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, _ = self.model_step(batch, batch_idx, "train")

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, decoded_segs, decoded_heatmaps, target_imgs = self.model_step(batch, batch_idx, "val")

        # TODO: compute the keypoint error
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, decoded_segs, decoded_heatmaps, target_imgs = self.model_step(
            batch, batch_idx, "test"
        )
        return loss

    def log_images(self, batch, **kwargs):
        inputs, targets = batch
        imgs = inputs["image"]
        target_imgs = targets["image"]
        outputs = self(imgs)
        decoded_segs = outputs["segs"]

        decoded_heatmaps = outputs["heatmaps"]

        input_images = imgs.detach().cpu()
        target_images = target_imgs.detach().cpu()
        target_segs = targets["segs"].detach().cpu()
        target_heatmaps = targets["heatmaps"].detach().cpu()
        target_keypoints = targets["keypoints"].detach().cpu()

        return dict(
            input_images=input_images,
            target_images=target_images,
            target_segs=target_segs,
            target_heatmaps=target_heatmaps,
            target_keypoints=target_keypoints,
            decoded_segs=decoded_segs,
            decoded_heatmaps=decoded_heatmaps,
        )
