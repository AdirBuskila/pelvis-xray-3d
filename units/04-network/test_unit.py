"""Checks for unit 04 (the network).

Every check_* takes YOUR function or class and tests it on small cases whose answer was worked out by
hand (output sizes, pixel values, parameter counts from the formula in the docstring) or on properties
(shapes, which voxels a pixel reaches). There is no reference solution in here. conv2d_naive is compared
with PyTorch's own convolution: matching it is the point of that TODO.

  notebook:  import test_unit as T;  T.check_conv_out_size(conv_out_size)
  pytest:    pytest units/04-network   (tests the copies you moved into src/xray3d/nets.py;
                                        anything not moved yet is skipped)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xray3d import nets
from xray3d.checks import call, close, run, skip_if_missing


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _make(cls, *args, **kwargs):
    """Instantiate a student class; a NotImplementedError in __init__ means 'still a TODO'."""
    return call(cls, *args, **kwargs)


def _fwd(model, x, grad=False):
    try:
        with torch.set_grad_enabled(grad):
            return model(x)
    except NotImplementedError as e:
        from xray3d.checks import TodoNotDone
        raise TodoNotDone(f"{type(model).__name__}.forward is still a TODO: {e}") from None


# ---------------------------------------------------------------- part A: convolution (video 09)

def check_conv_out_size(fn):
    def same():
        assert call(fn, 128, 3, 1, 1) == 128, "kernel 3, stride 1, padding 1 keeps the size (128 -> 128)"

    def halving():
        assert call(fn, 128, 3, 2, 1) == 64, "kernel 3, stride 2, padding 1 halves it (128 -> 64)"
        assert call(fn, 7, 3, 2, 1) == 4, "an odd size: 7 with kernel 3, stride 2, padding 1 gives 4 (positions 0, 2, 4, 6)"

    def no_padding():
        assert call(fn, 5, 3) == 3, "5 pixels, kernel 3, no padding: the kernel fits in 3 places"
        assert call(fn, 5, 3, 2, 0) == 2, "5 pixels, kernel 3, stride 2, no padding: starts at 0 and 2"
        assert call(fn, 6, 3, 2, 0) == 2, "6 pixels, kernel 3, stride 2: a third start (4) would stick out; it's dropped"

    def transposed():
        assert call(fn, 8, 2, 2, 0, transposed=True) == 16, "transposed, kernel 2, stride 2: 8 -> 16 (each input pixel stamps a 2x2 block)"
        assert call(fn, 4, 3, 2, 0, transposed=True) == 9, "transposed, kernel 3, stride 2: 4 -> 9 (the overlapping stamps of video 10)"
        assert call(fn, 8, 4, 2, 1, transposed=True) == 16, "transposed, kernel 4, stride 2, padding 1: 8 -> 16"

    def is_int():
        v = call(fn, 128, 3, 2, 1)
        assert isinstance(v, (int, np.integer)), f"a size is a whole number; got {type(v).__name__} {v}"

    run("conv_out_size", same, halving, no_padding, transposed, is_int)


class _NoTorchConv:
    """Inside this block PyTorch's own conv2d raises: conv2d_naive must do the sums itself."""

    def __enter__(self):
        def forbidden(*a, **k):
            raise AssertionError("conv2d_naive called PyTorch's conv2d; write the sliding window yourself")
        self.saved = (F.conv2d, torch.conv2d, nn.Conv2d.forward)
        F.conv2d = torch.conv2d = forbidden
        nn.Conv2d.forward = forbidden

    def __exit__(self, *exc):
        F.conv2d, torch.conv2d, nn.Conv2d.forward = self.saved


def check_conv2d_naive(fn):
    img = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)  # [[1,2,3],[4,5,6],[7,8,9]]

    def naive(*a, **k):
        with _NoTorchConv():
            return call(fn, *a, **k)

    def box_2x2():
        got = naive(img, torch.ones(1, 1, 2, 2))
        close(got, [[[[12, 16], [24, 28]]]], "2x2 box of ones on [[1,2,3],[4,5,6],[7,8,9]] (1+2+4+5 = 12, ...)")

    def no_flip():
        w = torch.tensor([[[[1.0, 0], [0, 0]]]])
        got = naive(img, w)
        close(got, [[[[1, 2], [4, 5]]]], "kernel [[1,0],[0,0]] picks the TOP-LEFT pixel of each window "
              "(deep learning's 'convolution' does not flip the kernel)")

    def padding():
        got = naive(img, torch.ones(1, 1, 3, 3), padding=1)
        close(got, [[[[12, 21, 16], [27, 45, 33], [24, 39, 28]]]], "3x3 box with padding=1 (zeros around the image)")

    def stride_bias_channels():
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, 7, 6, generator=g)
        w = torch.randn(4, 3, 3, 3, generator=g)
        b = torch.randn(4, generator=g)
        want = F.conv2d(x, w, b, stride=2, padding=1)
        got = naive(x, w, b, stride=2, padding=1)
        close(got, want, "batch 2, 3 -> 4 channels, 7x6 image, stride 2, padding 1, with bias, vs F.conv2d", atol=1e-4)

    run("conv2d_naive", box_2x2, no_flip, padding, stride_bias_channels)


def check_ConvBlock(cls):
    torch.manual_seed(0)

    def shapes_2d():
        y = _fwd(_make(cls, 3, 8), torch.randn(2, 3, 16, 16))
        assert tuple(y.shape) == (2, 8, 16, 16), f"ConvBlock(3, 8) on (2, 3, 16, 16) should give (2, 8, 16, 16), got {tuple(y.shape)}"
        y = _fwd(_make(cls, 3, 8, stride=2), torch.randn(2, 3, 16, 16))
        assert tuple(y.shape) == (2, 8, 8, 8), f"stride=2 should halve 16 -> 8, got {tuple(y.shape)}"

    def shapes_3d():
        y = _fwd(_make(cls, 3, 8, dim=3), torch.randn(2, 3, 8, 8, 8))
        assert tuple(y.shape) == (2, 8, 8, 8, 8), f"dim=3 on (2, 3, 8, 8, 8) should give (2, 8, 8, 8, 8), got {tuple(y.shape)}"
        y = _fwd(_make(cls, 3, 8, stride=2, dim=3), torch.randn(2, 3, 8, 8, 8))
        assert tuple(y.shape) == (2, 8, 4, 4, 4), f"dim=3, stride=2 should give (2, 8, 4, 4, 4), got {tuple(y.shape)}"

    def relu_last():
        y = _fwd(_make(cls, 3, 8), torch.randn(2, 3, 16, 16))
        assert float(y.min()) >= 0, "the block ends with a ReLU, so nothing it outputs is negative"

    def has_instancenorm():
        m = _make(cls, 3, 8)
        assert any(isinstance(s, nn.modules.instancenorm._InstanceNorm) for s in m.modules()), "the block needs an InstanceNorm layer"

    def parameter_count():
        # conv without bias: c_in * c_out * 3^dim weights; InstanceNorm(affine=True): a scale and a shift per channel
        n2 = _n_params(_make(cls, 3, 8))
        assert n2 == 3 * 8 * 9 + 2 * 8, (f"ConvBlock(3, 8) has {n2} parameters; expected 3*8*9 (3x3 kernels, no bias) + 2*8 "
                                         f"(the norm's scale and shift, affine=True) = {3 * 8 * 9 + 16}")
        n3 = _n_params(_make(cls, 3, 8, dim=3))
        assert n3 == 3 * 8 * 27 + 2 * 8, f"ConvBlock(3, 8, dim=3) has {n3} parameters; expected 3*8*27 + 2*8 = {3 * 8 * 27 + 16}"

    run("ConvBlock", shapes_2d, shapes_3d, relu_last, has_instancenorm, parameter_count)


# ---------------------------------------------------------------- part B: the network (video 10)

def encoder_params(c_in=1, widths=(16, 32, 64, 128), out=(4, 8, 16)) -> int:
    """Parameter count of Encoder2D as its docstring specifies it (ConvBlock = 3x3 conv without bias + InstanceNorm(affine=True))."""
    blk = lambda a, b: a * b * 9 + 2 * b
    total = blk(c_in, widths[0]) + blk(widths[0], widths[0])
    total += sum(blk(a, b) + blk(b, b) for a, b in zip(widths[:-1], widths[1:]))
    return total + sum(w * o + o for w, o in zip(widths[-len(out):], out))   # the 1x1 heads (with bias)


def decoder_params(c_in=(8, 16, 32), widths=(8, 16, 32)) -> int:
    """Parameter count of Decoder3D as its docstring specifies it."""
    blk = lambda a, b: a * b * 27 + 2 * b
    total = blk(c_in[-1], widths[-1]) + blk(widths[-1], widths[-1])         # at the coarsest level
    for lvl in range(len(widths) - 2, -1, -1):
        w = widths[lvl]
        total += widths[lvl + 1] * w * 8 + 2 * w                           # ConvTranspose3d(k=2, no bias) + InstanceNorm3d(affine=True)
        total += blk(w + c_in[lvl], w) + blk(w, w)                          # after the concat with the skip
    return total + widths[0] + 1                                            # 1x1x1 conv to one logit channel


def check_Encoder2D(cls):
    torch.manual_seed(0)

    def default_shapes():
        enc = _make(cls)
        ys = _fwd(enc, torch.randn(2, 1, 128, 128))
        got = [tuple(y.shape) for y in ys]
        want = [(2, 4, 64, 64), (2, 8, 32, 32), (2, 16, 16, 16)]
        assert got == want, f"Encoder2D() on (2, 1, 128, 128) should return maps of shapes {want} (fine to coarse), got {got}"
        assert tuple(getattr(enc, "out", ())) == (4, 8, 16), "set self.out = tuple(out) (XrayTo3D needs the channel counts)"

    def other_widths():
        ys = _fwd(_make(cls, 2, (8, 16, 32), (5, 6)), torch.randn(1, 2, 32, 32))
        got = [tuple(y.shape) for y in ys]
        assert got == [(1, 5, 16, 16), (1, 6, 8, 8)], (f"Encoder2D(2, (8, 16, 32), out=(5, 6)) on (1, 2, 32, 32): stages at 32, 16, 8; "
                                                       f"the last two through 1x1 convs to 5 and 6 channels -> [(1, 5, 16, 16), (1, 6, 8, 8)]; got {got}")

    def parameter_count():
        n = _n_params(_make(cls))
        want = encoder_params()
        assert n == want, (f"Encoder2D() has {n:,} parameters, expected {want:,}: stage 0 = two ConvBlocks at full size, then per "
                           "stage ConvBlock(a, b, stride=2) + ConvBlock(b, b), and a 1x1 Conv2d (with bias) per returned map")

    run("Encoder2D", default_shapes, other_widths, parameter_count)


def check_lift_view(fn):
    n, B, C = 4, 2, 3

    def lit(view, row, col):
        f = torch.zeros(B, C, n, n)
        f[1, 2, row, col] = 1.0
        v = call(fn, f, view)
        assert tuple(v.shape) == (B, C, n, n, n), f"lift_view of (2, 3, 4, 4) should be (2, 3, 4, 4, 4), got {tuple(v.shape)}"
        v = v.detach().numpy()
        assert v[0].sum() == 0 and v[1, :2].sum() == 0, "the lit pixel was in batch 1, channel 2; other batches/channels must stay 0"
        return {tuple(int(a) for a in ijk) for ijk in np.argwhere(v[1, 2] > 0.5)}, v[1, 2]

    def ap_pixel():
        # AP image: rows head -> feet, columns patient's Right -> Left; the ray runs along A (array axis j)
        got, v = lit("ap", 1, 0)
        want = {(3, j, 2) for j in range(n)}
        assert got == want, (f"AP pixel (row 1, col 0) is near the head and at the patient's far RIGHT: it should light "
                             f"i = 3 (Right is +i), k = 2 (row 1 from the top), every j (the ray). It lit {sorted(got)}")
        close(v[3, :, 2], [1, 1, 1, 1], "the AP pixel's value copied along its ray")

    def ap_pixel_2():
        got, _ = lit("ap", 3, 2)
        want = {(1, j, 0) for j in range(n)}
        assert got == want, f"AP pixel (row 3 = feet, col 2) should light i = 1, k = 0, every j; it lit {sorted(got)}"

    def lat_pixel():
        # LAT image: rows head -> feet, columns Anterior -> Posterior; the ray runs along R (array axis i)
        got, _ = lit("lat", 0, 1)
        want = {(i, 2, 3) for i in range(n)}
        assert got == want, (f"LAT pixel (row 0 = head, col 1 = near the front) should light j = 2 (Anterior is +j), "
                             f"k = 3, every i (the ray); it lit {sorted(got)}")

    def lat_pixel_2():
        got, _ = lit("lat", 2, 3)
        want = {(i, 0, 1) for i in range(n)}
        assert got == want, f"LAT pixel (row 2, col 3 = most posterior) should light j = 0, k = 1, every i; it lit {sorted(got)}"

    def gradient():
        f = torch.randn(B, C, n, n, requires_grad=True)
        call(fn, f, "ap").sum().backward()
        close(f.grad, torch.full((B, C, n, n), float(n)), "each pixel is copied n times, so d(sum)/d(pixel) = n")

    run("lift_view", ap_pixel, ap_pixel_2, lat_pixel, lat_pixel_2, gradient)


def check_Decoder3D(cls):
    torch.manual_seed(0)
    vols = lambda B, cs, n: [torch.randn(B, c, n // 2 ** i, n // 2 ** i, n // 2 ** i) for i, c in enumerate(cs)]

    def default_shape():
        y = _fwd(_make(cls), vols(2, (8, 16, 32), 64))
        assert tuple(y.shape) == (2, 1, 64, 64, 64), f"Decoder3D() on volumes of 8 x 64^3, 16 x 32^3, 32 x 16^3 should give (2, 1, 64, 64, 64), got {tuple(y.shape)}"

    def other_widths():
        y = _fwd(_make(cls, (3, 5), (4, 6)), vols(1, (3, 5), 16))
        assert tuple(y.shape) == (1, 1, 16, 16, 16), f"Decoder3D((3, 5), (4, 6)) on 3 x 16^3 and 5 x 8^3 should give (1, 1, 16, 16, 16); got {tuple(y.shape)}"

    def logits():
        y = _fwd(_make(cls).eval(), vols(1, (8, 16, 32), 64))
        assert float(y.min()) < 0 and float(y.std()) > 0, "the output is logits (any real number): no ReLU or sigmoid after the last conv"

    def transposed():
        assert any(isinstance(m, nn.ConvTranspose3d) for m in _make(cls).modules()), "upsample with nn.ConvTranspose3d"

    def uses_skips():
        m = _make(cls).eval()
        v = vols(1, (8, 16, 32), 64)
        a = _fwd(m, v)
        v[0] = torch.zeros_like(v[0])
        b = _fwd(m, v)
        assert not torch.allclose(a, b), "changing the finest volume (the 64^3 skip) didn't change the output: concatenate it in"

    def prior_bias():
        # InstanceNorm maps an all-zero channel to 0 (+ its shift, which starts at 0), and no conv before the head has a bias:
        # a blank input reaches the head as zeros, so the output is exactly the head's bias
        y = _fwd(_make(cls).eval(), [torch.zeros_like(v) for v in vols(1, (8, 16, 32), 64)])
        close(y.mean(), -4.0, "a blank input should give logit -4 everywhere (start by predicting p = 0.018: bone is rare)", atol=1e-4)

    def parameter_count():
        n = _n_params(_make(cls))
        want = decoder_params()
        assert n == want, (f"Decoder3D() has {n:,} parameters, expected {want:,}: two ConvBlocks at the coarsest level; per finer level "
                           "ConvTranspose3d(k=2, s=2, no bias) + InstanceNorm3d + ReLU, concat the skip, two ConvBlocks; "
                           "then a 1x1x1 Conv3d to 1 channel")

    run("Decoder3D", default_shape, other_widths, logits, transposed, uses_skips, prior_bias, parameter_count)


def _saliency(model, x, view_idx, voxel):
    """|d logit(voxel) / d image| for one view: which pixels does that voxel depend on?"""
    x = x.clone().requires_grad_(True)
    out = _fwd(model, x, grad=True)
    out[0, 0][voxel].backward()
    return x.grad[0, view_idx].abs().numpy()


def check_XrayTo3D(cls):
    def shapes():
        torch.manual_seed(0)
        for nv in (1, 2):
            for fuse in ("sum", "concat"):
                y = _fwd(_make(cls, nv, fuse=fuse), torch.randn(2, nv, 128, 128))
                assert tuple(y.shape) == (2, 1, 64, 64, 64), f"XrayTo3D({nv}, fuse='{fuse}') should output (2, 1, 64, 64, 64), got {tuple(y.shape)}"

    def parameter_counts():
        e = encoder_params()
        cases = {(1, "concat"): e + decoder_params((4, 8, 16)), (2, "concat"): 2 * e + decoder_params((8, 16, 32)),
                 (2, "sum"): 2 * e + decoder_params((4, 8, 16))}
        for (nv, fuse), want in cases.items():
            n = _n_params(_make(cls, nv, fuse=fuse))
            assert n == want, (f"XrayTo3D({nv}, fuse='{fuse}') has {n:,} parameters, expected {want:,}: one Encoder2D per view "
                               f"({e:,} each), and a Decoder3D whose c_in is the encoder's out (sum) or out x n_views (concat)")

    def ap_orientation():
        torch.manual_seed(0)
        m = _make(cls, 2).eval()
        x = torch.randn(1, 2, 128, 128)
        g = _saliency(m, x, 0, (63, 32, 0))   # a voxel at the patient's far Right, at the feet
        right_feet, left_head = g[96:, :32].mean(), g[:32, 96:].mean()
        assert right_feet > 10 * left_head, ("the voxel (i=63 Right, k=0 feet) should depend on the BOTTOM-LEFT of the AP image "
                                             "(rows run head -> feet, columns Right -> Left), not on the top-right; "
                                             "check lift_view's flips")

    def lat_orientation():
        torch.manual_seed(0)
        m = _make(cls, 2).eval()
        x = torch.randn(1, 2, 128, 128)
        g = _saliency(m, x, 1, (32, 0, 63))   # a voxel at the back (Posterior), at the head
        back_head, front_feet = g[:32, 96:].mean(), g[96:, :32].mean()
        assert back_head > 10 * front_feet, ("the voxel (j=0 Posterior, k=63 head) should depend on the TOP-RIGHT of the LAT image "
                                             "(columns run Anterior -> Posterior), not on the bottom-left")

    def dropout():
        torch.manual_seed(0)
        m = _make(cls, 2, dropout=0.5)
        x = torch.randn(1, 2, 128, 128)
        m.train()
        with torch.no_grad():
            a, b = _fwd(m, x), _fwd(m, x)
        assert not torch.allclose(a, b), "with dropout=0.5 two forward passes in train mode should differ (unit 07's MC dropout)"
        m.eval()
        with torch.no_grad():
            a, b = _fwd(m, x), _fwd(m, x)
        assert torch.allclose(a, b), "in eval mode dropout is off: two forward passes must be identical"

    run("XrayTo3D", shapes, parameter_counts, ap_orientation, lat_orientation, dropout)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d/nets.py

def test_conv_out_size():
    check_conv_out_size(skip_if_missing(nets, "conv_out_size"))


def test_conv2d_naive():
    check_conv2d_naive(skip_if_missing(nets, "conv2d_naive"))


def test_ConvBlock():
    check_ConvBlock(skip_if_missing(nets, "ConvBlock"))


def test_Encoder2D():
    check_Encoder2D(skip_if_missing(nets, "Encoder2D"))


def test_lift_view():
    check_lift_view(skip_if_missing(nets, "lift_view"))


def test_Decoder3D():
    check_Decoder3D(skip_if_missing(nets, "Decoder3D"))


def test_XrayTo3D():
    check_XrayTo3D(skip_if_missing(nets, "XrayTo3D"))
