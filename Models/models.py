import timm
import torch
import torch.nn as nn

class CXRModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 14,
        vit_model: nn.Module = None,
        model_weights: dict | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # 1. Build or take backbone WITHOUT classifier head
        if vit_model is not None:
            backbone = vit_model
        else:
            raise ValueError("vit_model must be provided.")

        # If the backbone still has a classifier head, strip/reset it
        if hasattr(backbone, "reset_classifier"):
            backbone.reset_classifier(0)
        elif hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        # 2. Load pretrained backbone weights (ignore classifier mismatches)
        if model_weights is not None:
            missing, unexpected = backbone.load_state_dict(
                model_weights, strict=False
            )
            if len(unexpected) > 0:
                print(f"[CXRModel] Ignored unexpected keys in state_dict: {unexpected}")
            if len(missing) > 0:
                print(f"[CXRModel] Missing keys when loading state_dict: {missing}")

        self.backbone = backbone

        # 3. Freeze backbone params for linear probing
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 4. New trainable linear head
        in_dim = getattr(self.backbone, "num_features", None)
        if in_dim is None:
            # Some ViTs only expose embed_dim
            in_dim = getattr(self.backbone, "embed_dim")

        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # For timm ViT with num_classes=0, backbone(x) returns pooled features [B, C]
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
