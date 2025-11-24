import timm
import torch
import torch.nn as nn
from torchvision import models
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

class MAECXRModel(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

    def forward(self, x):
        """
        Encoder forward for downstream tasks: 
        no masking, use full image, return CLS token representation.
        """
        # embed patches
        x = self.patch_embed(x)                           # [N, L, D]
        x = x + self.pos_embed[:, 1:, :]                  # add pos embed (no cls yet)

        # add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # [N, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)             # [N, 1+L, D]

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                  # [N, 1+L, D]

        # return CLS embedding
        return x[:, 0]                                    # [N, D]

class MIMICCXREncoder(nn.Module):
    def __init__(self):
        """
        Initialize the MIMICCXREncoder, which encodes chest X-ray (CXR) images using
        a modified ResNet-50 architecture.

        If `args.pretrained` is True, the ResNet-50 model is initialized with
        pre-trained weights from the ImageNet dataset ("IMAGENET1K_V2"). The
        fully connected layer (fc) of ResNet-50 is replaced with a new Linear
        layer to match the desired output dimensionality (`args.d`). A LayerNorm
        layer is added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
        super().__init__()

        self.resnet = models.resnet50(weights=None)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 8192, bias=True)

        self.layer_norm = nn.LayerNorm(8192)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): CXR data (batch_sz, 3, 320, 320).
        Returns:
            x (torch.Tensor): learned CXR representation (batch_sz, d)
        """
        x = self.resnet(x)
        x = self.layer_norm(x)
        return x

class CXRModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: str = "imagenet", # imagenet/mae/mimic/pacx
        backbone_name: str = "vit_base_patch16_224",
        model_checkpoints: str | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # --------- 1. Build backbone inside the class ----------
        self.mode = mode
        self.backbone_name = backbone_name
        self.model_checkpoints = model_checkpoints
        self.backbone = self._build_backbone()

        # --------- 2. Freeze backbone if linear probing ----------
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # --------- 3. New classifier head ----------
             # Get output dim dynamically
        dummy_in = torch.randn(1, 3, 224, 224, device=next(self.backbone.parameters()).device)
        with torch.no_grad():
            out = self.backbone(dummy_in)
            in_dim = out.shape[1]
        
        self.head = nn.Linear(in_dim, num_classes)

    def _build_backbone(self):
        if self.mode == "imagenet":
            backbone = timm.create_model(
                self.backbone_name,
                pretrained=True,
                num_classes=0,   # feature extractor
            )
        elif self.mode == "mae":
            # Load the checkpoint
            ckpt = torch.load(
                self.model_checkpoints,
                map_location="cpu",
                weights_only=False
            )

            state = ckpt["state_dict"]
            encoder_state = {k: v for k, v in state.items() if k.startswith("model.") and not k.startswith("model.decoder")}

            encoder_state_stripped = {}
            for k, v in encoder_state.items():
                new_key = k.replace("model.", "")   # Encoder expects keys without the "model." prefix
                encoder_state_stripped[new_key] = v
            
            # ViT Base backbone
            backbone = MAECXRModel(
                        patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    )
            
            missing, unexpected = backbone.load_state_dict(encoder_state_stripped, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            
        elif self.mode == "mimic":
            # Load the checkpoint
            ckpt = torch.load(
                self.model_checkpoints,
                map_location="cpu",
                weights_only=False
            )

            state = ckpt["state_dict"]

            # ---- 1. Filter CXR encoder keys ----
            cxr_state = {k: v for k, v in state.items() if k.startswith("cxr_encoder.")}

            # ---- 2. Strip the "cxr_encoder." prefix ----
            cxr_state_stripped = {}
            for k, v in cxr_state.items():
                new_key = k.replace("cxr_encoder.", "")   # CXREncoder expects keys starting with "resnet."
                cxr_state_stripped[new_key] = v

            # ---- 3. Load NON-STRICT into MIMICCXREncoder ----
            backbone = MIMICCXREncoder()
            missing, unexpected = backbone.load_state_dict(cxr_state_stripped, strict=False)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        elif self.mode == "pacx":
            raise NotImplementedError("PACX backbone loading not implemented yet.")
        else:
            raise ValueError(f"Unknown model mode: {self.mode}. Supported modes are: imagenet, mae, mimic, pacx.")
        
        return backbone

    def forward(self, x):
        feats = self.backbone(x)     # [B, C]
        logits = self.head(feats)    # [B, num_classes]
        return logits
