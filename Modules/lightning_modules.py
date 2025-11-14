import pytorch_lightning as pl

# For plotting validation reconstructions
import torchvision.utils as vutils
import torch
import pytorch_lightning as pl
from torch.optim import AdamW

from Utils.utils import build_param_groups
from Models.mae import vit_base_patch16, vit_large_patch16, vit_huge_patch14

class MAELightningModule(pl.LightningModule):
    def __init__(self,
                 size = "base",
                 norm_pix_loss=False,
                 mask_ratio: float = 0.75,
                 lr: float = 1.5e-4,
                 weight_decay: float = 0.05,
                 betas: tuple = (0.9, 0.95),
                 warmup_epochs: int = 10,
                 log_images_every_n_epochs: int = 10,
                 log_max_images: int = 8):
        super().__init__()
        self.save_hyperparameters()

        if size == "base":
            self.model = vit_base_patch16(norm_pix_loss=norm_pix_loss)
        elif size == "large":
            self.model = vit_large_patch16(norm_pix_loss=norm_pix_loss)
        elif size == "huge":
            self.model = vit_huge_patch14(norm_pix_loss=norm_pix_loss)
        else:
            raise ValueError(f"Unknown MAE size: {size}")

    def forward_loss(self, imgs, pred, mask):
        target = self.model.patchify(imgs)
        if self.hparams.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        loss = (pred - target).pow(2).mean(dim=-1)               # [N, L]
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)         # masked-only
        return loss

    def forward(self, imgs):
        return self.model(imgs, mask_ratio=self.hparams.mask_ratio)

    def training_step(self, batch, batch_idx):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        _, pred, mask = self(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        _, pred, mask = self(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.size(0), sync_dist=True)

        # optional: log reconstructions on first batch every N epochs
        if (
            batch_idx == 0
            and self.hparams.log_images_every_n_epochs > 0
            and (self.current_epoch % self.hparams.log_images_every_n_epochs == 0)
        ):
            self._log_reconstructions(imgs, pred)
        return loss

    # ----- helpers -----
    @torch.no_grad()
    def _log_reconstructions(self, imgs: torch.Tensor, pred: torch.Tensor, tag: str = "reconstructions"):
        """
        Log a grid: top = ground truth, bottom = MAE reconstruction.
        Works with TensorBoard or Weights & Biases if attached as Lightning loggers.

        Args:
            imgs: [B,3,H,W] input batch in normalized space (e.g., ImageNet stats).
            pred: MAE decoder outputs in the same normalized space as the training target.
            tag:  image tag/name for the logger.
        """
        if imgs.ndim != 4 or imgs.size(1) != 3:
            return  # nothing to do; only handle RGB 4D tensors

        # MAE outputs normalized pixels; clamp gently to avoid extreme values in display
        rec = self.model.unpatchify(pred).clamp(-3, 3)

        n = int(getattr(self.hparams, "log_max_images", 8))
        n = max(1, min(n, imgs.size(0)))

        # de-normalization toward [0,1]; keep on same device then cast/move right before grid
        def denorm(x: torch.Tensor) -> torch.Tensor:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x * std + mean).clamp(0, 1)

        gt_vis  = denorm(imgs[:n])
        rec_vis = denorm(rec[:n])

        # stack into two rows: first row GT, second row REC
        grid_src = torch.cat([gt_vis, rec_vis], dim=0).detach().float().cpu()
        # ensure uint8 for image loggers when converting to numpy
        grid = vutils.make_grid(grid_src, nrow=n, padding=2, pad_value=0.5)

        logger = getattr(self, "logger", None)
        if logger is None or getattr(logger, "experiment", None) is None:
            return

        exp = logger.experiment
        step = int(getattr(self, "global_step", 0))

        # TensorBoard
        if hasattr(exp, "add_image"):
            exp.add_image(tag, grid, global_step=step)
            return

        # Weights & Biases
        if hasattr(exp, "log"):
            try:
                import numpy as np
                np_img = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
                import wandb
                exp.log({tag: wandb.Image(np_img)}, step=step)
            except Exception:
                pass

    def configure_optimizers(self):
        optimizer = AdamW(
            build_param_groups(self.model, self.hparams.weight_decay),
            lr=self.hparams.lr, betas=self.hparams.betas
        )

        # Warmup + cosine to zero across epochs
        # Use LambdaLR so T_max = total_epochs is implicit
        def lr_lambda(current_epoch):
            # linear warmup
            if current_epoch < self.hparams.warmup_epochs:
                return float(current_epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            # cosine
            total = max(self.trainer.max_epochs - self.hparams.warmup_epochs, 1)
            progress = (current_epoch - self.hparams.warmup_epochs) / total
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}