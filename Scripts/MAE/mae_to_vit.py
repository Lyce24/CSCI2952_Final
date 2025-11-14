import torch
from typing import Dict
from timm.models.layers import trunc_normal_

# from mae import mae_vit_base_patch16_dec512d8b, mae_vit_large_patch16_dec512d8b, mae_vit_huge_patch14_dec512d8b
from vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def get_vit_from_mae(pretrained_model: Dict, global_pool: bool = False) -> torch.nn.Module:
    """
    Retrieves a ViT model for fine-tuning based on a pretrained MAE model.

    Args:
        pretrained_model (Dict): Pretrained model state dictionary.
        global_pool (bool, optional): Flag indicating whether to use global pooling for fine-tuning.
                                      Defaults to False.

    Returns:
        torch.nn.Module: Model for fine-tuning.

    """
    state_dict = pretrained_model["state_dict"]

    # Find the key containing the search string 'patch_embed.proj.weight'
    found_key = None
    search_string = 'patch_embed.proj.weight'
    for key in state_dict.keys():
        if search_string in key:
            found_key = key
            break

    # Get model size and input channel count from the found key
    model_size, in_chans, _, _ = state_dict[found_key].shape

    if "_orig_mod" in found_key:
        compiled = True
        replace_string = 'model._orig_mod.'
    else:
        compiled = False
        replace_string = 'model.'

    # Select the appropriate ViT model based on the model size
    if model_size == 768:
        vit = vit_base_patch16(in_chans=in_chans)
    elif model_size == 1024:
        vit = vit_large_patch16(in_chans=in_chans)
    elif model_size == 1280:
        vit = vit_huge_patch14(in_chans=in_chans)

    pretrained_dict = {}
    for key in state_dict.keys():
        new_key = key.replace(replace_string, '')
        pretrained_dict[new_key] = state_dict[key]

    finetune_state_dict = vit.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in pretrained_dict and pretrained_dict[k].shape != finetune_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del pretrained_dict[k]

    # Interpolate position embedding
    interpolate_pos_embed(vit, pretrained_dict)

    # Load pre-trained model
    msg = vit.load_state_dict(pretrained_dict, strict=False)
    print(msg)

    if global_pool:  # Using global pooling for fine-tuning and don't use for fine-tuning
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # Manually initialize fc layer: following MoCo v3
    trunc_normal_(vit.head.weight, std=0.01)

    # For linear prob only
    # Hack: revise model's head with BN
    vit.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(vit.head.in_features, affine=False, eps=1e-6),
        vit.head
    )

    # Freeze all but the head
    for _, p in vit.named_parameters():
        p.requires_grad = False
    for _, p in vit.head.named_parameters():
        p.requires_grad = True

    return vit

if __name__ == "__main__":
    mae = torch.load("checkpoint.pt")
    vit = get_vit_from_mae(mae, global_pool=False)