"""FP8 fake-quantization utilities for OpenVLA inference."""

import gc
from typing import Iterable, Tuple

import torch
import torch.nn as nn


DEFAULT_FP8_SKIP_KEYWORDS = ("lm_head", "output")
DEFAULT_FP8_DTYPE = "e4m3fn"


def resolve_fp8_dtype(fp8_dtype: str) -> torch.dtype:
    normalized = fp8_dtype.lower().replace("float8_", "")
    dtype_map = {
        "e4m3fn": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
        "e4m3fnuz": torch.float8_e4m3fnuz,
        "e5m2fnuz": torch.float8_e5m2fnuz,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}. Choose one of: {', '.join(dtype_map)}")
    return dtype_map[normalized]


@torch.no_grad()
def _quantize_to_fp8(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(fp8_dtype)
    tensor_fp32 = tensor.to(torch.float32)
    amax = tensor_fp32.abs().amax()
    scale = (amax / finfo.max).clamp_min(1e-12)
    tensor_scaled = (tensor_fp32 / scale).clamp(min=finfo.min, max=finfo.max)
    return tensor_scaled.to(fp8_dtype), scale


def _fp8_dequantize(tensor_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor_fp8.to(dtype) * scale.to(dtype)


class FP8Linear(nn.Module):
    """`nn.Linear` wrapper with persistent FP8 weights and optional activation fake quantization."""

    def __init__(
        self,
        original_linear: nn.Linear,
        fp8_dtype: str = DEFAULT_FP8_DTYPE,
        quantize_activations: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.fp8_dtype_name = fp8_dtype
        self.fp8_dtype = resolve_fp8_dtype(fp8_dtype)
        self.quantize_activations = quantize_activations
        self.compute_dtype = original_linear.weight.dtype

        weight_fp8, weight_scale = _quantize_to_fp8(original_linear.weight.detach(), self.fp8_dtype)
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("weight_scale", weight_scale)

        if original_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                original_linear.bias.detach().clone(),
                requires_grad=original_linear.bias.requires_grad,
            )

    def _quant_dequant_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activations:
            return x
        x_fp8, x_scale = _quantize_to_fp8(x, self.fp8_dtype)
        return _fp8_dequantize(x_fp8, x_scale, x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_fp8.device != x.device:
            self.to(x.device)

        x_quant = self._quant_dequant_activation(x)
        weight = _fp8_dequantize(self.weight_fp8, self.weight_scale, x_quant.dtype)
        bias = self.bias.to(x_quant.dtype) if self.bias is not None else None
        return nn.functional.linear(x_quant, weight, bias)


def replace_fp8_layers_recursive(
    module: nn.Module,
    fp8_dtype: str = DEFAULT_FP8_DTYPE,
    quantize_activations: bool = True,
    skip_keywords: Iterable[str] = DEFAULT_FP8_SKIP_KEYWORDS,
    prefix: str = "",
) -> int:
    """Recursively replace `nn.Linear` modules with `FP8Linear` wrappers."""
    count = 0
    skip_keywords = tuple(skip_keywords)

    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if isinstance(child, FP8Linear):
            continue

        if isinstance(child, nn.Linear):
            if any(keyword in fullname for keyword in skip_keywords):
                continue

            setattr(
                module,
                name,
                FP8Linear(
                    child,
                    fp8_dtype=fp8_dtype,
                    quantize_activations=quantize_activations,
                ),
            )
            count += 1
        else:
            count += replace_fp8_layers_recursive(
                child,
                fp8_dtype=fp8_dtype,
                quantize_activations=quantize_activations,
                skip_keywords=skip_keywords,
                prefix=fullname,
            )

    return count


def apply_fp8(
    module: nn.Module,
    fp8_dtype: str = DEFAULT_FP8_DTYPE,
    quantize_activations: bool = True,
    device: torch.device | str | None = None,
    skip_keywords: Tuple[str, ...] = DEFAULT_FP8_SKIP_KEYWORDS,
    module_label: str = "module",
) -> nn.Module:
    """Apply FP8 Linear replacement for inference-time memory experiments."""
    resolve_fp8_dtype(fp8_dtype)
    was_training = module.training

    if device is not None:
        module = module.to(device)

    replaced = replace_fp8_layers_recursive(
        module,
        fp8_dtype=fp8_dtype,
        quantize_activations=quantize_activations,
        skip_keywords=skip_keywords,
    )

    if device is not None:
        module = module.to(device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not was_training:
        module.eval()

    mode = "W8A8 fake quant" if quantize_activations else "W8 only"
    print(f"[FP8] {module_label}: replaced {replaced} Linear layer(s), dtype={fp8_dtype}, mode={mode}.")
    return module
