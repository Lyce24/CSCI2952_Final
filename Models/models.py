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
        unfreeze_backbone: bool = False,
        task: str = "sl", # sl, seg
    ):
        super().__init__()

        # --------- 1. Build backbone inside the class ----------
        self.mode = mode
        self.backbone_name = backbone_name
        self.model_checkpoints = model_checkpoints
        self.task = task
        self.backbone = self._build_backbone()
        self.h = 224 // 16   # hardcoded for 224 and patch_size=16
        self.w = 224 // 16
        
        # --------- 2. Freeze backbone if linear probing ----------
        if not unfreeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # 3. Classifier head, infer feature dim with dummy forward
        #    (backbone is on CPU by default here)
        device = next(self.backbone.parameters()).device
        dummy_in = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = self.backbone(dummy_in)
            assert out.ndim == 2, f"Backbone output should be [B, C], got {out.shape}"
            in_dim = out.shape[1]
        
        if self.task == "sl":
            self.head = nn.Linear(in_dim, num_classes)
        elif self.task == "seg":
            self.decoder = nn.Sequential(
                nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            )
            
    def _build_backbone(self):
        if self.mode == "imagenet":
            backbone = timm.create_model(
                self.backbone_name,
                pretrained=True,
                num_classes=0,   # feature extractor
            )

            # Remove classifier head if any
            if hasattr(backbone, "reset_classifier"):
                backbone.reset_classifier(0)
            elif hasattr(backbone, "head"):
                backbone.head = nn.Identity()
                
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
            if self.task != "sl":
                raise ValueError("MIMIC backbone currently only supports 'sl' task.")
            
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
        if self.task != "sl":
            raise ValueError("forward called but task is not 'sl'")
        logits = self.head(feats)    # [B, num_classes]
        return logits

    # --------------------
    # ViT token encoder
    # --------------------
    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Re-implement ViT forward to get all patch tokens.

        Returns:
            tokens: [B, L, D] (patch tokens only, CLS dropped)
        """
        patch_embed = self.backbone.patch_embed
        cls_token = self.backbone.cls_token      # [1,1,D]
        pos_embed = self.backbone.pos_embed      # [1, 1+L, D]
        blocks = self.backbone.blocks
        norm = self.backbone.norm

        # 1) patch embedding
        x = patch_embed(x)                       # [B, L, D]
        # 2) add positional embeddings (skip CLS pos)
        x = x + pos_embed[:, 1:, :]              # [B, L, D]

        # 3) prepend CLS token with its pos embedding
        cls_tok = cls_token + pos_embed[:, :1, :]      # [1,1,D]
        cls_tok = cls_tok.expand(x.shape[0], -1, -1)   # [B,1,D]
        x = torch.cat((cls_tok, x), dim=1)             # [B,1+L,D]

        # 4) transformer blocks
        for blk in blocks:
            x = blk(x)
        x = norm(x)                             # [B,1+L,D]

        # 5) drop CLS → patch tokens
        tokens = x[:, 1:, :]                    # [B,L,D]
        return tokens

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224]
        return: feature map [B, D, H', W'] where H'=W'=img_size/patch_size.
        """
        tokens = self._encode_tokens(x)         # [B, L, D]
        B, L, D = tokens.shape

        expected_L = self.h * self.w
        if L != expected_L:
            raise ValueError(f"Expected {expected_L} patches (h={self.h}, w={self.w}), got {L}")

        feat = tokens.transpose(1, 2).contiguous().view(B, D, self.h, self.w)
        return feat
    
    def seg_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Segmentation forward pass.
        x: [B, 3, 224, 224]
        return: seg map [B, 1, 224, 224]
        """
        feat = self._encode(x)                  # [B, D, H', W']
        if self.task != "seg":
            raise ValueError("seg_forward called but task is not 'seg'")
        seg_map = self.decoder(feat)            # [B, 1, 224, 224]
        return seg_map