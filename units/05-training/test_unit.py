"""Checks for unit 05 (training).

Every check_* takes YOUR function and tests it on tiny tensors whose answer was worked out by hand
(a sigmoid of 0 is 0.5, ln 2 = 0.6931, ...), or on properties (did the weights move the way one SGD step
moves them?). There is no reference solution in here.

  notebook:  import test_unit as T;  T.check_soft_dice_loss(soft_dice_loss)
  pytest:    pytest units/05-training   (tests the copies you moved into src/xray3d/losses.py and training.py;
                                         anything not moved yet is skipped)
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from xray3d import losses, training
from xray3d.checks import call, close, run, skip_if_missing

LN2 = math.log(2.0)


def _t(values, shape=(1, 1, 1, 2, 2)):  # (B, C, D, H, W), as the docstrings promise
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _scalar(x, what):
    assert isinstance(x, torch.Tensor), f"{what}: return a torch tensor (so .backward() works), got {type(x).__name__}"
    assert x.dim() == 0, f"{what}: the loss should be one number (a 0-dim tensor), got shape {tuple(x.shape)}"
    return float(x)


# ---------------------------------------------------------------- part A: the loss

def check_soft_dice_loss(fn):
    target = _t([1, 0, 0, 0])

    def half():
        # sigmoid(0) = 0.5 everywhere: overlap 0.5, sum of p = 2, sum of t = 1 -> (2*0.5 + 1) / (2 + 1 + 1) = 0.5
        close(_scalar(call(fn, torch.zeros(1, 1, 1, 2, 2), target), "soft_dice_loss"), 0.5, "logits 0 everywhere, one voxel of four is bone")

    def no_smoothing():
        close(float(call(fn, torch.zeros(1, 1, 1, 2, 2), target, eps=0.0)), 1 - 1 / 3, "the same with eps=0: 1 - 2*0.5/3")

    def perfect():
        t = (torch.arange(64).reshape(1, 1, 4, 4, 4) % 5 == 0).float()
        close(float(call(fn, 30 * (2 * t - 1), t)), 0.0, "a confident, correct prediction", atol=1e-4)

    def empty_and_empty():
        close(float(call(fn, torch.full((1, 1, 1, 2, 2), -30.0), torch.zeros(1, 1, 1, 2, 2))), 0.0,
              "nothing to find and nothing predicted: that's right (eps makes 0/0 into 1/1)", atol=1e-4)

    def all_missed():
        close(float(call(fn, torch.full((1, 1, 1, 2, 2), -30.0), torch.ones(1, 1, 1, 2, 2))), 0.8,
              "four bone voxels, none predicted: 1 - (0 + 1)/(0 + 4 + 1)", atol=1e-4)

    def per_sample_mean():
        logits = torch.stack([torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 30.0)])
        t = torch.stack([target[0], torch.ones(1, 1, 2, 2)])
        close(float(call(fn, logits, t)), 0.25, "batch of 2: losses 0.5 and 0.0 -> their mean 0.25. (One Dice over the whole "
              "batch would give 0.167: compute it per sample, then average)", atol=1e-4)

    def differentiable():
        logits = torch.zeros(1, 1, 1, 2, 2, requires_grad=True)
        call(fn, logits, target).backward()
        g = logits.grad
        assert g is not None and float(g.abs().sum()) > 0, "no gradient reached the logits: use sigmoid (smooth), not a threshold"
        assert float(g[0, 0, 0, 0, 0]) < 0 < float(g[0, 0, 0, 1, 1]), "raising the logit of the bone voxel should lower the loss, raising an empty one should raise it"

    run("soft_dice_loss", half, no_smoothing, perfect, empty_and_empty, all_missed, per_sample_mean, differentiable)


def check_bce_dice_loss(fn):
    target = _t([1, 0, 0, 0])

    def value():
        close(_scalar(call(fn, torch.zeros(1, 1, 1, 2, 2), target), "bce_dice_loss"), LN2 + 0.5,
              "logits 0: BCE = ln 2 = 0.6931 on every voxel, soft Dice loss 0.5")

    def weights():
        close(float(call(fn, torch.zeros(1, 1, 1, 2, 2), target, w_bce=0.0)), 0.5, "w_bce=0 leaves the Dice part")
        close(float(call(fn, torch.zeros(1, 1, 1, 2, 2), target, w_dice=0.0)), LN2, "w_dice=0 leaves the BCE part")
        close(float(call(fn, torch.zeros(1, 1, 1, 2, 2), target, w_bce=2.0, w_dice=3.0)), 2 * LN2 + 1.5, "w_bce=2, w_dice=3")

    def confident_and_wrong():
        # true BCE of logit 50 for an empty voxel is log(1 + e^50) = 50. sigmoid(50) rounds to exactly 1.0 in float32,
        # so sigmoid-then-log gives log(0) (PyTorch clamps it to 100). The "with logits" version stays exact.
        got = float(call(fn, torch.full((1, 1, 1, 1, 1), 50.0), torch.zeros(1, 1, 1, 1, 1)))
        close(got, 50.0 + 0.5, "one empty voxel predicted with logit 50: BCE 50 + Dice loss 0.5 "
              "(use F.binary_cross_entropy_with_logits; sigmoid then log loses it)", atol=1e-3)

    def differentiable():
        logits = torch.zeros(1, 1, 1, 2, 2, requires_grad=True)
        call(fn, logits, target).backward()
        assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, "no gradient reached the logits"

    run("bce_dice_loss", value, weights, confident_and_wrong, differentiable)


# ---------------------------------------------------------------- part B: one step, and the validation Dice

def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))


def _tiny_batch():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(6, 3, generator=g)
    return x, x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.3   # a target the tiny model can learn


def check_train_step(fn):
    mse = nn.functional.mse_loss

    def one_sgd_step():
        model = _tiny_model().eval()               # left in eval mode on purpose: train_step must switch it
        ref = copy.deepcopy(model).train()
        x, y = _tiny_batch()
        # what one SGD step (lr 0.1) must do, computed here by hand
        ref_loss = mse(ref(x), y)
        grads = torch.autograd.grad(ref_loss, list(ref.parameters()))
        want = [p.detach() - 0.1 * g for p, g in zip(ref.parameters(), grads)]
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for p in model.parameters():                # stale gradients from "the previous step"
            p.grad = torch.full_like(p, 5.0)
        got_loss = call(fn, model, (x, y), opt, mse, "cpu")
        assert isinstance(got_loss, float), f"return the loss as a plain Python float (loss.item()), got {type(got_loss).__name__}"
        close(got_loss, ref_loss.item(), "the returned loss is the loss BEFORE the step", atol=1e-5)
        assert model.training, "call model.train() first (BatchNorm and dropout behave differently in training)"
        for (name, p), w in zip(model.named_parameters(), want):
            if not torch.allclose(p.detach(), w, atol=1e-5):
                raise AssertionError(f"parameter {name} is not what one SGD step gives. Did you zero the old gradients "
                                     "(optimizer.zero_grad()) before loss.backward(), and call optimizer.step()?")

    def batch_as_list():
        model = _tiny_model()
        x, y = _tiny_batch()
        loss = call(fn, model, [x, y], torch.optim.SGD(model.parameters(), lr=0.1), mse, "cpu")
        assert np.isfinite(loss), "a batch given as a list [x, y] should work too (that's what a DataLoader yields)"

    def loss_goes_down():
        model = _tiny_model()
        x, y = _tiny_batch()
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        ls = [call(fn, model, (x, y), opt, mse, "cpu") for _ in range(60)]
        assert ls[-1] < 0.5 * ls[0], f"60 steps on the same batch should at least halve the loss: {ls[0]:.3f} -> {ls[-1]:.3f}"

    run("train_step", one_sgd_step, batch_as_list, loss_goes_down)


class _LogitsAreInput(nn.Module):
    """A fake network whose output is its input: the 'images' in these batches ARE the logits."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled()))
        return x * self.w


def _vol(on, value=5.0, n=4):
    """A (1, 4, 4, 4) volume: `value` at the flat indices in `on`, minus `value` elsewhere."""
    v = torch.full((n ** 3,), -abs(value))
    v[list(on)] = value
    return v.reshape(1, n, n, n)


def _mask(on, n=4):
    m = torch.zeros(n ** 3)
    m[list(on)] = 1
    return m.reshape(1, n, n, n)


def check_validation_dice(fn):
    s1 = (_vol([0, 1, 2, 3]), _mask([0, 1, 2, 3]))            # perfect: Dice 1
    s2 = (_vol([2, 3, 4, 5], 0.3), _mask([0, 1, 2, 3]))       # 2 of 4 overlap: Dice 2*2/(4+4) = 0.5 (logit 0.3 = p 0.57: bone)
    s3 = (_vol([]), _mask([]))                                # nothing there, nothing predicted: Dice 1
    s4 = (_vol([7]), _mask([]))                               # nothing there, one voxel predicted: Dice 0
    batch = lambda *ss: (torch.stack([s[0] for s in ss]), torch.stack([s[1] for s in ss]))

    def per_sample():
        model = _LogitsAreInput()
        got = call(fn, model, [batch(s1, s2), batch(s3)], "cpu")
        assert isinstance(got, float), f"return a plain float, got {type(got).__name__}"
        close(got, (1 + 0.5 + 1) / 3, "three samples with Dice 1, 0.5, 1 -> mean 0.833 (the mean over SAMPLES; "
              "averaging the two batches' means would give 0.875). Logit > 0 means p > 0.5")

    def empty_cases():
        close(call(fn, _LogitsAreInput(), [batch(s3)], "cpu"), 1.0, "empty target, empty prediction: Dice 1 by definition")
        close(call(fn, _LogitsAreInput(), [batch(s4)], "cpu"), 0.0, "empty target, one voxel predicted: Dice 0")

    def eval_mode_no_grad():
        model = _LogitsAreInput().train()
        call(fn, model, [batch(s1)], "cpu")
        assert model.calls, "validation_dice never called the model"
        training_flag, grad_on = model.calls[-1]
        assert not training_flag, "call model.eval() first (BatchNorm must use its running averages, dropout must be off)"
        assert not grad_on, "wrap the loop in torch.no_grad(): no backward pass here, so don't keep the graph"

    run("validation_dice", per_sample, empty_cases, eval_mode_no_grad)


def check_overfit(history: dict):
    """The overfit-one-batch test: the loss must collapse and the batch's Dice must pass 0.9."""
    def loss_collapsed():
        ls = history["loss"]
        assert ls[-1] < 0.3 * ls[0], f"the loss only went from {ls[0]:.3f} to {ls[-1]:.3f}; on one batch it should collapse"

    def dice_high():
        d = history["dice"][-1]
        assert d > 0.9, (f"Dice on the batch it trained on is {d:.3f}; a network that can't memorise one batch has a bug "
                         "(look at the loss, the learning rate, the orientation of lift_view)")

    run("overfit one batch", loss_collapsed, dice_high)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d

def test_soft_dice_loss():
    check_soft_dice_loss(skip_if_missing(losses, "soft_dice_loss"))


def test_bce_dice_loss():
    check_bce_dice_loss(skip_if_missing(losses, "bce_dice_loss"))


def test_train_step():
    check_train_step(skip_if_missing(training, "train_step"))


def test_validation_dice():
    check_validation_dice(skip_if_missing(training, "validation_dice"))
