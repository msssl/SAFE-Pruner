def draw_patches_overlay_attention(image_primary, image_wrist, last_layer_attention_avg_last_tok_image, patch_size=14, alpha=0.4):
    """
    Draws attention-based colored overlays with 4-tier color scheme.
    Patches are divided into 4 tiers based on token count (quartiles),
    each tier has a distinct color, and within each tier, brightness indicates relative attention.
    
    Tier 1 (bottom 25% tokens): Blue - Low attention
    Tier 2 (25-50% tokens): Green - Medium-low attention
    Tier 3 (50-75% tokens): Yellow - Medium-high attention
    Tier 4 (top 25% tokens): Red - High attention
    """
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    
    if isinstance(last_layer_attention_avg_last_tok_image, torch.Tensor):
        attention_weights = last_layer_attention_avg_last_tok_image.detach().cpu().float().numpy()
    else:
        attention_weights = last_layer_attention_avg_last_tok_image
    
    # Normalize attention weights to [0, 1]
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    # Calculate quartiles (25th, 50th, 75th percentiles) for tier boundaries
    q25 = np.percentile(attention_weights, 25)
    q50 = np.percentile(attention_weights, 50)
    q75 = np.percentile(attention_weights, 75)
    
    num_patches_per_image = len(attention_weights) // 2
    attn_primary = attention_weights[:num_patches_per_image]
    attn_wrist = attention_weights[num_patches_per_image:]
    
    def get_tier_color(attn_value, q25, q50, q75):
        """
        Returns RGB color based on attention value tier (determined by quartiles).
        Within each tier, brightness increases with attention value.
        """
        # Determine which tier (0-3) based on quartiles
        if attn_value < q25:
            tier = 0  # Blue - bottom 25%
            tier_min, tier_max = 0, q25
        elif attn_value < q50:
            tier = 1  # Green - 25-50%
            tier_min, tier_max = q25, q50
        elif attn_value < q75:
            tier = 2  # Yellow - 50-75%
            tier_min, tier_max = q50, q75
        else:
            tier = 3  # Red - top 25%
            tier_min, tier_max = q75, 1.0
        
        # Calculate local_intensity within tier (0 to 1)
        if tier_max > tier_min:
            local_intensity = (attn_value - tier_min) / (tier_max - tier_min)
        else:
            local_intensity = 0.5
        
        # Map local_intensity to brightness: 0.3 (dark) to 1.0 (bright)
        brightness = 0.3 + 0.7 * local_intensity
        
        # Define base colors for each tier
        if tier == 0:  # Blue - low attention
            r, g, b = int(30 * brightness), int(60 * brightness), int(255 * brightness)
        elif tier == 1:  # Green - medium-low attention
            r, g, b = int(30 * brightness), int(200 * brightness), int(60 * brightness)
        elif tier == 2:  # Yellow - medium-high attention
            r, g, b = int(255 * brightness), int(220 * brightness), int(30 * brightness)
        else:  # Red - high attention
            r, g, b = int(255 * brightness), int(50 * brightness), int(30 * brightness)
        
        return (r, g, b)
    
    def apply_attention_overlay(image, attention, q25, q50, q75):
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        num_patches = int(np.sqrt(len(attention)))
        
        for pid, attn_value in enumerate(attention):
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            
            # Get color based on tier and intensity
            r, g, b = get_tier_color(attn_value, q25, q50, q75)
            
            draw.rectangle([top_left, bottom_right], fill=(r, g, b, int(255 * alpha)))
        
        return Image.alpha_composite(image, overlay).convert("RGB")
    
    vis_primary = apply_attention_overlay(image_primary, attn_primary, q25, q50, q75)
    vis_wrist = apply_attention_overlay(image_wrist, attn_wrist, q25, q50, q75)
    
    return vis_primary, vis_wrist

def draw_patches_overlay_attention_every_image(image_primary, image_wrist, last_layer_attention_avg_last_tok_image, patch_size=14, alpha=0.4):
    """
    Draws attention-based colored overlays with 4-tier color scheme.
    Patches are divided into 4 tiers based on token count (quartiles),
    each tier has a distinct color, and within each tier, brightness indicates relative attention.
    
    Tier 1 (bottom 25% tokens): Blue - Low attention
    Tier 2 (25-50% tokens): Green - Medium-low attention
    Tier 3 (50-75% tokens): Yellow - Medium-high attention
    Tier 4 (top 25% tokens): Red - High attention
    """
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    
    if isinstance(last_layer_attention_avg_last_tok_image, torch.Tensor):
        attention_weights = last_layer_attention_avg_last_tok_image.detach().cpu().float().numpy()
    else:
        attention_weights = last_layer_attention_avg_last_tok_image
    
    # Normalize attention weights to [0, 1]
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    num_patches_per_image = len(attention_weights) // 2
    attn_primary = attention_weights[:num_patches_per_image]
    attn_wrist = attention_weights[num_patches_per_image:]
    
    def get_tier_color(attn_value, q25, q50, q75):
        """
        Returns RGB color based on attention value tier (determined by quartiles).
        Within each tier, brightness increases with attention value.
        """
        # Determine which tier (0-3) based on quartiles
        if attn_value < q25:
            tier = 0  # Blue - bottom 25%
            tier_min, tier_max = 0, q25
        elif attn_value < q50:
            tier = 1  # Green - 25-50%
            tier_min, tier_max = q25, q50
        elif attn_value < q75:
            tier = 2  # Yellow - 50-75%
            tier_min, tier_max = q50, q75
        else:
            tier = 3  # Red - top 25%
            tier_min, tier_max = q75, 1.0
        
        # Calculate local_intensity within tier (0 to 1)
        if tier_max > tier_min:
            local_intensity = (attn_value - tier_min) / (tier_max - tier_min)
        else:
            local_intensity = 0.5
        
        # Map local_intensity to brightness: 0.3 (dark) to 1.0 (bright)
        brightness = 0.3 + 0.7 * local_intensity
        
        # Define base colors for each tier
        if tier == 0:  # Blue - low attention
            r, g, b = int(30 * brightness), int(60 * brightness), int(255 * brightness)
        elif tier == 1:  # Green - medium-low attention
            r, g, b = int(30 * brightness), int(200 * brightness), int(60 * brightness)
        elif tier == 2:  # Yellow - medium-high attention
            r, g, b = int(255 * brightness), int(220 * brightness), int(30 * brightness)
        else:  # Red - high attention
            r, g, b = int(255 * brightness), int(50 * brightness), int(30 * brightness)
        
        return (r, g, b)
    
    def apply_attention_overlay(image, attention, q25, q50, q75):
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        num_patches = int(np.sqrt(len(attention)))
        
        for pid, attn_value in enumerate(attention):
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            
            # Get color based on tier and intensity
            r, g, b = get_tier_color(attn_value, q25, q50, q75)
            
            draw.rectangle([top_left, bottom_right], fill=(r, g, b, int(255 * alpha)))
        
        return Image.alpha_composite(image, overlay).convert("RGB")

    # Calculate quartiles (25th, 50th, 75th percentiles) for tier boundaries
    q25 = np.percentile(attn_primary, 25)
    q50 = np.percentile(attn_primary, 50)
    q75 = np.percentile(attn_primary, 75)
    vis_primary = apply_attention_overlay(image_primary, attn_primary, q25, q50, q75)

    q25 = np.percentile(attn_wrist, 25)
    q50 = np.percentile(attn_wrist, 50)
    q75 = np.percentile(attn_wrist, 75)
    vis_wrist = apply_attention_overlay(image_wrist, attn_wrist, q25, q50, q75)
    
    return vis_primary, vis_wrist

def draw_patches_overlay_attention_every_image_3tier(
    image_primary,
    image_wrist,
    last_layer_attention_avg_last_tok_image,
    patch_size=14,
    alpha=0.4,
):
    """
    3-tier overlay:
    - Top 25% important tokens
    - 25%~50% important tokens
    - 50%~100% important tokens
    """
    import torch
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(last_layer_attention_avg_last_tok_image, torch.Tensor):
        attention_weights = last_layer_attention_avg_last_tok_image.detach().cpu().float().numpy()
    else:
        attention_weights = np.asarray(last_layer_attention_avg_last_tok_image, dtype=np.float32)

    # normalize to [0, 1]
    attn_min, attn_max = attention_weights.min(), attention_weights.max()
    attention_weights = (attention_weights - attn_min) / (attn_max - attn_min + 1e-8)

    num_patches_per_image = len(attention_weights) // 2
    attn_primary = attention_weights[:num_patches_per_image]
    attn_wrist = attention_weights[num_patches_per_image:]

    # tier 1: 25~50%            -> yellow
    # tier 2: 50~100%           -> green
    TIER_COLORS = {
        0: (230, 130, 50),
        1: (90, 45, 25),
        2: (90, 45, 25),
    }

    def assign_tiers_by_rank(attn):
        n = len(attn)
        order = np.argsort(-attn)  # descending, larger = more important
        k25 = max(1, int(np.ceil(0.25 * n)))
        k50 = max(k25, int(np.ceil(0.50 * n)))

        tiers = np.full(n, 2, dtype=np.int32)         # default: 50~100%
        tiers[order[:k25]] = 0                        # top 25%
        tiers[order[k25:k50]] = 1                     # 25~50%
        return tiers

    def apply_attention_overlay_3tier(image, attention):
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        tiers = assign_tiers_by_rank(attention)
        num_patches = int(np.sqrt(len(attention)))

        for pid in range(len(attention)):
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)

            r, g, b = TIER_COLORS[int(tiers[pid])]
            draw.rectangle([top_left, bottom_right], fill=(r, g, b, int(255 * alpha)))

        return Image.alpha_composite(image, overlay).convert("RGB")

    vis_primary = apply_attention_overlay_3tier(image_primary, attn_primary)
    vis_wrist = apply_attention_overlay_3tier(image_wrist, attn_wrist)

    return vis_primary, vis_wrist

def draw_patches_overlay_attention_every_image_3tier_tier0_only(
    image_primary,
    image_wrist,
    last_layer_attention_avg_last_tok_image,
    patch_size=14,
    alpha=0.4,
):
    """
    3-tier ranking (every image), but only tier_0 is colored:
    - tier_0: top 25% important tokens -> colored
    - tier_1: 25%~50% -> no overlay
    - tier_2: 50%~100% -> no overlay
    """
    import torch
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(last_layer_attention_avg_last_tok_image, torch.Tensor):
        attention_weights = last_layer_attention_avg_last_tok_image.detach().cpu().float().numpy()
    else:
        attention_weights = np.asarray(last_layer_attention_avg_last_tok_image, dtype=np.float32)

    # normalize to [0, 1]
    attn_min, attn_max = attention_weights.min(), attention_weights.max()
    attention_weights = (attention_weights - attn_min) / (attn_max - attn_min + 1e-8)

    num_patches_per_image = len(attention_weights) // 2
    attn_primary = attention_weights[:num_patches_per_image]
    attn_wrist = attention_weights[num_patches_per_image:]

    TIER0_COLOR = (230, 130, 50)

    def assign_tiers_by_rank(attn):
        n = len(attn)
        order = np.argsort(-attn)  # descending, larger = more important
        k25 = max(1, int(np.ceil(0.25 * n)))
        k50 = max(k25, int(np.ceil(0.50 * n)))

        tiers = np.full(n, 2, dtype=np.int32)  # default: tier_2
        tiers[order[:k25]] = 0                 # top 25% -> tier_0
        tiers[order[k25:k50]] = 1              # 25~50% -> tier_1
        return tiers

    def apply_attention_overlay_tier0_only(image, attention):
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        tiers = assign_tiers_by_rank(attention)
        num_patches = int(np.sqrt(len(attention)))

        for pid in range(len(attention)):
            if int(tiers[pid]) != 0:
                continue

            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            r, g, b = TIER0_COLOR
            draw.rectangle([top_left, bottom_right], fill=(r, g, b, int(255 * alpha)))

        return Image.alpha_composite(image, overlay).convert("RGB")

    vis_primary = apply_attention_overlay_tier0_only(image_primary, attn_primary)
    vis_wrist = apply_attention_overlay_tier0_only(image_wrist, attn_wrist)

    return vis_primary, vis_wrist
