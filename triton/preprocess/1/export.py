from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _min(a: float, b: float) -> float:
    """
    Return the smaller of two floats.

    Args:
        a: First value.
        b: Second value.

    Returns:
        The minimum of ``a`` and ``b``.
    """
    return a if a < b else b


class Letterbox640(nn.Module):
    """
    Letterbox and normalize images to 3x640x640 FP32 on CUDA.

    Args:
        target_hw: Target height and width as ``(H, W)``.
    """

    def __init__(self, target_hw: Tuple[int, int] = (640, 640)) -> None:
        super().__init__()
        self.th = int(target_hw[0])
        self.tw = int(target_hw[1])

        self.register_buffer(
            "scale_factor", torch.tensor(1.0 / 255.0, dtype=torch.float32)
        )
        self.register_buffer(
            "pad_value", torch.tensor(114.0 / 255.0, dtype=torch.float32)
        )

    @torch.jit.export
    def _process_one(self, img_nhwc_u8: torch.Tensor) -> torch.Tensor:
        """
        Process a single HWC uint8 image to CHW FP32 640x640 on CUDA.

        Args:
            img_nhwc_u8: Input tensor with shape ``[H, W, 3]`` and dtype ``uint8`` on CUDA or CPU.

        Returns:
            Tensor with shape ``[3, 640, 640]`` and dtype ``float32`` on CUDA.
        """
        x = (
            img_nhwc_u8.to(device="cuda", dtype=torch.float32)
            * self.scale_factor
        )

        H = int(x.shape[0])
        W = int(x.shape[1])

        h_ratio = float(self.th) / float(H)
        w_ratio = float(self.tw) / float(W)
        scale = _min(h_ratio, w_ratio)

        new_h = int(round(float(H) * scale))
        new_w = int(round(float(W) * scale))

        x = x.permute(2, 0, 1).contiguous()

        x = F.interpolate(
            x.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_w_total = self.tw - new_w
        pad_h_total = self.th - new_h
        pad_w_left = pad_w_total // 2
        pad_w_right = pad_w_total - pad_w_left
        pad_h_top = pad_h_total // 2
        pad_h_bot = pad_h_total - pad_h_top

        x = F.pad(
            x,
            (pad_w_left, pad_w_right, pad_h_top, pad_h_bot),
            mode="constant",
            value=float(self.pad_value),
        )

        return x.to(dtype=torch.float32, device="cuda")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Batch-aware forward to letterbox and normalize images.

        Args:
            images: Either ``[H, W, 3]`` or ``[N, H, W, 3]`` with dtype ``uint8``.

        Returns:
            If input is batched: tensor of shape ``[N, 3, 640, 640]`` FP32 on CUDA.
            If single image: tensor of shape ``[1, 3, 640, 640]`` FP32 on CUDA.
        """
        if images.dim() == 3:
            out = self._process_one(images)
            return out.unsqueeze(0)
        else:
            N = int(images.shape[0])
            outs = []
            for i in range(N):
                outs.append(self._process_one(images[i]))
            return torch.stack(outs, dim=0)


def build_model(device: str = "cuda") -> torch.jit.ScriptModule:
    """
    Script and return the Letterbox640 model.

    Args:
        device: Target device string, e.g., ``"cuda"`` or ``"cpu"``.

    Returns:
        A ``torch.jit.ScriptModule`` of ``Letterbox640`` placed on CUDA.
    """
    m = Letterbox640((640, 640)).eval()
    if device != "cuda":
        print(
            "[WARN] This model moves inputs to CUDA internally; using CUDA is recommended."
        )
    m = m.to("cuda")
    scripted = torch.jit.script(m)
    return scripted


def make_example_input(device: str = "cuda") -> torch.Tensor:
    """
    Create an example NHWC uint8 batch input.

    Args:
        device: Target device string for the tensor allocation.

    Returns:
        Tensor with shape ``(2, 720, 1280, 3)`` and dtype ``uint8`` on the requested device.
    """
    return torch.randint(
        0,
        256,
        (2, 720, 1280, 3),
        dtype=torch.uint8,
        device="cuda" if device == "cuda" else "cpu",
    )


def script_and_save(
    out_path: str, device: str = "cuda", verify: bool = False
) -> None:
    """
    Script the model, optionally verify, and save to disk.

    Args:
        out_path: Destination path for the serialized TorchScript file.
        device: Target device string for example input generation.
        verify: If ``True``, prints output shapes, dtype, and device.
    """
    scripted = build_model(device=device)
    with torch.inference_mode():
        dummy = make_example_input(device=device)
        out = scripted(dummy)

        if verify:
            print(
                f"[verify] output.shape = {tuple(out.shape)}, dtype = {out.dtype}, device = {out.device}"
            )

    scripted.save(out_path)
    print(f"[done] saved TorchScript to: {out_path}")


def main() -> None:
    """
    Entrypoint to script and save the model to ``./model.pt`` with verification.
    """
    script_and_save(out_path="./test_model.pt", verify=True)


if __name__ == "__main__":
    main()
