import logging
import math
import os
import random
import time
from collections import namedtuple

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import webdataset as wds
from einops import rearrange
from torch import Tensor
from torchvision import models
from torchvision.transforms import GaussianBlur
from transformers import get_cosine_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import wandb

# ---------------------------------------------------------------------------- #
#                                      AE                                      #
# ---------------------------------------------------------------------------- #


def swish(x) -> Tensor:
    return x * torch.sigmoid(x)


StandardizedC2d = nn.Conv2d


class FP32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.head_dim = 64
        self.num_heads = in_channels // self.head_dim
        self.norm = FP32GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.qkv = StandardizedC2d(
            in_channels, in_channels * 3, kernel_size=1, bias=False
        )
        self.proj_out = StandardizedC2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )
        nn.init.normal_(self.proj_out.weight, std=0.2 / math.sqrt(in_channels))

    def attention(self, h_) -> Tensor:
        h_ = self.norm(h_)
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)
        b, c, h, w = q.shape
        q = rearrange(
            q, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        k = rearrange(
            k, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        v = rearrange(
            v, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return h_

    def forward(self, x) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = FP32GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = StandardizedC2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = FP32GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = StandardizedC2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = StandardizedC2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

        # init conv2 as very small number
        nn.init.normal_(self.conv2.weight, std=0.0001 / self.out_channels)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):

        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = StandardizedC2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = StandardizedC2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        use_attn: bool = True,
        use_wavelet: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_wavelet = use_wavelet
        if self.use_wavelet:
            self.wavelet_transform = wavelet_transform_multi_channel
            self.conv_in = StandardizedC2d(
                4 * in_channels, self.ch * 2, kernel_size=3, stride=1, padding=1
            )
            ch_mult[0] *= 2
        else:
            self.wavelet_transform = nn.Identity()
            self.conv_in = StandardizedC2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1
            )

        curr_res = resolution
        in_ch_mult = (2 if self.use_wavelet else 1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1 and not (
                self.use_wavelet and i_level == 0
            ):
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in) if use_attn else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = FP32GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = StandardizedC2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )
        for module in self.modules():
            if isinstance(module, StandardizedC2d):
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GroupNorm):
                nn.init.zeros_(module.bias)

    def forward(self, x) -> Tensor:
        h = self.wavelet_transform(x)
        h = self.conv_in(h)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1 and not (
                self.use_wavelet and i_level == 0
            ):
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        use_attn: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.conv_in = StandardizedC2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in) if use_attn else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = FP32GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = StandardizedC2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

        # initialize all bias to zero
        for module in self.modules():
            if isinstance(module, StandardizedC2d):
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GroupNorm):
                nn.init.zeros_(module.bias)

    def forward(self, z) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z) -> Tensor:
        mean = z
        if self.sample:
            std = 0.00
            return mean * (1 + std * torch.randn_like(mean))
        else:
            return mean


class VAE(nn.Module):
    def __init__(
        self,
        resolution,
        in_channels,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        use_attn,
        decoder_also_perform_hr,
        use_wavelet,
    ):
        super().__init__()
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            use_attn=use_attn,
            use_wavelet=use_wavelet,
        )
        self.decoder = Decoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult + [4] if decoder_also_perform_hr else ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            use_attn=use_attn,
        )
        self.reg = DiagonalGaussian()

    def forward(self, x) -> Tensor:
        z = self.encoder(x)
        z_s = self.reg(z)
        decz = self.decoder(z_s)
        return decz, z


# ---------------------------------------------------------------------------- #
#                                     UTILS                                     #
# ---------------------------------------------------------------------------- #


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        try:
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))
        except:
            print("Failed to load vgg.pth, downloading...")
            os.system(
                "wget https://github.com/richzhang/PerceptualSimilarity/raw/refs/heads/master/lpips/weights/v0.1/vgg.pth"
            )
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))

        self.load_state_dict(
            data,
            strict=False,
        )

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = models.vgg16(pretrained=True)

        self.slice1 = nn.Sequential(_vgg.features[:4])
        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice5 = nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier5[-1].weight)

    def forward(self, x):
        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features2 = self.slice2(features1)
        features3 = self.slice3(features2)
        features4 = self.slice4(features3)
        features5 = self.slice5(features4)

        bc1 = self.binary_classifier1(features1).flatten(1)
        bc2 = self.binary_classifier2(features2).flatten(1)
        bc3 = self.binary_classifier3(features3).flatten(1)
        bc4 = self.binary_classifier4(features4).flatten(1)
        bc5 = self.binary_classifier5(features5).flatten(1)

        return bc1 + bc2 + bc3 + bc4 + bc5


dec_lo, dec_hi = (
    torch.Tensor([-0.1768, 0.3536, 1.0607, 0.3536, -0.1768, 0.0000]),
    torch.Tensor([0.0000, -0.0000, 0.3536, -0.7071, 0.3536, -0.0000]),
)

filters = torch.stack(
    [
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
    ],
    dim=0,
)

filters_expanded = filters.unsqueeze(1)


def prepare_filter(device):
    global filters_expanded
    filters_expanded = filters_expanded.to(device)


def wavelet_transform_multi_channel(x, levels=4):
    B, C, H, W = x.shape
    padded = torch.nn.functional.pad(x, (2, 2, 2, 2))

    # use predefined filters
    global filters_expanded

    ress = []
    for ch in range(C):
        res = torch.nn.functional.conv2d(
            padded[:, ch : ch + 1], filters_expanded, stride=2
        )
        ress.append(res)

    res = torch.cat(ress, dim=1)
    H_out, W_out = res.shape[2], res.shape[3]
    res = res.view(B, C, 4, H_out, W_out)
    res = res.view(B, 4 * C, H_out, W_out)
    return res


# ---------------------------------------------------------------------------- #
#                                 VAE TRAINER                                  #
# ---------------------------------------------------------------------------- #


def gan_disc_loss(real_preds, fake_preds, disc_type="bce"):
    if disc_type == "bce":
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            real_preds, torch.ones_like(real_preds)
        )
        fake_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_preds, torch.zeros_like(fake_preds)
        )
        # eval its online performance
        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

    if disc_type == "hinge":
        real_loss = nn.functional.relu(1 - real_preds).mean()
        fake_loss = nn.functional.relu(1 + fake_preds).mean()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

    return (real_loss + fake_loss) * 0.5, avg_real_preds, avg_fake_preds, acc


MAX_WIDTH = 256

this_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.CenterCrop(256),
        transforms.Resize(MAX_WIDTH),
    ]
)


def this_transform_random_crop_resize(x, width=MAX_WIDTH):

    x = transforms.ToTensor()(x)
    x = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)

    if random.random() < 0.5:
        x = transforms.RandomCrop(width)(x)
    else:
        x = transforms.Resize(width)(x)
        x = transforms.RandomCrop(width)(x)

    return x


def create_dataloader(url, batch_size, num_workers, do_shuffle=True, just_resize=False):
    dataset = wds.WebDataset(
        url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker
    )
    dataset = dataset.shuffle(1000) if do_shuffle else dataset

    dataset = (
        dataset.decode("rgb")
        .to_tuple("jpg;png")
        .map_tuple(
            this_transform_random_crop_resize if not just_resize else this_transform
        )
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def blurriness_heatmap(input_image):
    grayscale_image = input_image.mean(dim=1, keepdim=True)

    laplacian_kernel = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, -20, 1, 1],
            [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    laplacian_kernel = laplacian_kernel.view(1, 1, 5, 5)

    laplacian_kernel = laplacian_kernel.to(input_image.device)

    edge_response = F.conv2d(grayscale_image, laplacian_kernel, padding=2)

    edge_magnitude = GaussianBlur(kernel_size=(13, 13), sigma=(2.0, 2.0))(
        edge_response.abs()
    )

    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (
        edge_magnitude.max() - edge_magnitude.min() + 1e-8
    )

    blurriness_map = 1 - edge_magnitude

    blurriness_map = torch.where(
        blurriness_map < 0.8, torch.zeros_like(blurriness_map), blurriness_map
    )

    return blurriness_map.repeat(1, 3, 1, 1)


def vae_loss_function(x, x_reconstructed, z, do_pool=True, do_recon=False):
    # downsample images by factor of 8
    if do_recon:
        if do_pool:
            x_reconstructed_down = F.interpolate(
                x_reconstructed, scale_factor=1 / 16, mode="area"
            )
            x_down = F.interpolate(x, scale_factor=1 / 16, mode="area")
            recon_loss = ((x_reconstructed_down - x_down)).abs().mean()
        else:
            x_reconstructed_down = x_reconstructed
            x_down = x

            recon_loss = (
                ((x_reconstructed_down - x_down) * blurriness_heatmap(x_down))
                .abs()
                .mean()
            )
            recon_loss_item = recon_loss.item()
    else:
        recon_loss = 0
        recon_loss_item = 0

    elewise_mean_loss = z.pow(2)
    zloss = elewise_mean_loss.mean()

    with torch.no_grad():
        actual_mean_loss = elewise_mean_loss.mean()
        actual_ks_loss = actual_mean_loss.mean()

    vae_loss = recon_loss * 0.0 + zloss * 0.1
    return vae_loss, {
        "recon_loss": recon_loss_item,
        "kl_loss": actual_ks_loss.item(),
        "average_of_abs_z": z.abs().mean().item(),
        "std_of_abs_z": z.abs().std().item(),
        "average_of_logvar": 0.0,
        "std_of_logvar": 0.0,
    }


def cleanup():
    dist.destroy_process_group()


@click.command()
@click.option(
    "--dataset_url", type=str, default="", help="URL for the training dataset"
)
@click.option(
    "--test_dataset_url", type=str, default="", help="URL for the test dataset"
)
@click.option("--num_epochs", type=int, default=2, help="Number of training epochs")
@click.option("--batch_size", type=int, default=1, help="Batch size for training")
@click.option("--do_ganloss", is_flag=True, help="Whether to use GAN loss")
@click.option(
    "--learning_rate_vae", type=float, default=1e-5, help="Learning rate for VAE"
)
@click.option(
    "--learning_rate_disc",
    type=float,
    default=2e-4,
    help="Learning rate for discriminator",
)
@click.option("--vae_resolution", type=int, default=256, help="Resolution for VAE")
@click.option("--vae_in_channels", type=int, default=3, help="Input channels for VAE")
@click.option("--vae_ch", type=int, default=128, help="Base channel size for VAE")
@click.option(
    "--vae_ch_mult", type=str, default="1,2,2,4", help="Channel multipliers for VAE"
)
@click.option(
    "--vae_num_res_blocks",
    type=int,
    default=2,
    help="Number of residual blocks for VAE",
)
@click.option(
    "--vae_z_channels", type=int, default=16, help="Number of latent channels for VAE"
)
@click.option("--run_name", type=str, default="run", help="Name of the run for wandb")
@click.option(
    "--max_steps", type=int, default=1000, help="Maximum number of steps to train for"
)
@click.option(
    "--evaluate_every_n_steps", type=int, default=250, help="Evaluate every n steps"
)
@click.option("--load_path", type=str, default=None, help="Path to load the model from")
@click.option("--do_clamp", is_flag=True, help="Whether to clamp the latent codes")
@click.option(
    "--clamp_th", type=float, default=8.0, help="Clamp threshold for the latent codes"
)
@click.option(
    "--max_spatial_dim",
    type=int,
    default=256,
    help="Maximum spatial dimension for overall training",
)
@click.option(
    "--do_attn", type=bool, default=False, help="Whether to use attention in the VAE"
)
@click.option(
    "--decoder_also_perform_hr",
    type=bool,
    default=False,
    help="Whether to perform HR decoding in the decoder",
)
@click.option(
    "--project_name",
    type=str,
    default="vae_sweep_attn_lr_width",
    help="Project name for wandb",
)
@click.option(
    "--crop_invariance",
    type=bool,
    default=False,
    help="Whether to perform crop invariance",
)
@click.option(
    "--flip_invariance",
    type=bool,
    default=False,
    help="Whether to perform flip invariance",
)
@click.option(
    "--do_compile",
    type=bool,
    default=False,
    help="Whether to compile the model",
)
@click.option(
    "--use_wavelet",
    type=bool,
    default=False,
    help="Whether to use wavelet transform in the encoder",
)
@click.option(
    "--augment_before_perceptual_loss",
    type=bool,
    default=False,
    help="Whether to augment the images before the perceptual loss",
)
@click.option(
    "--downscale_factor",
    type=int,
    default=32,
    help="Downscale factor for the latent space",
)
@click.option(
    "--use_lecam",
    type=bool,
    default=False,
    help="Whether to use Lecam",
)
@click.option(
    "--disc_type",
    type=str,
    default="bce",
    help="Discriminator type",
)
def train(
    dataset_url,
    test_dataset_url,
    num_epochs,
    batch_size,
    do_ganloss,
    learning_rate_vae,
    learning_rate_disc,
    vae_resolution,
    vae_in_channels,
    vae_ch,
    vae_ch_mult,
    vae_num_res_blocks,
    vae_z_channels,
    run_name,
    max_steps,
    evaluate_every_n_steps,
    load_path,
    do_clamp,
    clamp_th,
    max_spatial_dim,
    do_attn,
    decoder_also_perform_hr,
    project_name,
    crop_invariance,
    flip_invariance,
    do_compile,
    use_wavelet,
    augment_before_perceptual_loss,
    downscale_factor,
    use_lecam,
    disc_type,
):

    # fix random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    start_train = 0
    end_train = 58

    test = end_train + 1

    dataset_url = f"/home/azazelle/AI/Optimizer_Arena/data/mscoco/{{{start_train:05d}..{end_train:05d}}}.tar"
    test_dataset_url = f"/home/azazelle/AI/Optimizer_Arena/data/mscoco/{test:05d}.tar"

    device = "cuda" if torch.cuda.is_available() else "cpu" # simplified device selection
    print(f"using device: {device}")

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "learning_rate_vae": learning_rate_vae,
            "learning_rate_disc": learning_rate_disc,
            "vae_ch": vae_ch,
            "vae_resolution": vae_resolution,
            "vae_in_channels": vae_in_channels,
            "vae_ch_mult": vae_ch_mult,
            "vae_num_res_blocks": vae_num_res_blocks,
            "vae_z_channels": vae_z_channels,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "do_ganloss": do_ganloss,
            "do_attn": do_attn,
            "use_wavelet": use_wavelet,
        },
    )

    vae = VAE(
        resolution=vae_resolution,
        in_channels=vae_in_channels,
        ch=vae_ch,
        out_ch=vae_in_channels,
        ch_mult=[int(x) for x in vae_ch_mult.split(",")],
        num_res_blocks=vae_num_res_blocks,
        z_channels=vae_z_channels,
        use_attn=do_attn,
        decoder_also_perform_hr=decoder_also_perform_hr,
        use_wavelet=use_wavelet,
    ).cuda()

    discriminator = PatchDiscriminator().cuda()
    discriminator.requires_grad_(True)

    prepare_filter(device)

    if do_compile:
        vae.encoder = torch.compile(
            vae.encoder, fullgraph=False, mode="max-autotune"
        )
        vae.decoder = torch.compile(
            vae.decoder, fullgraph=False, mode="max-autotune"
        )

    # context
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    optimizer_G = optim.AdamW(
        [
            {
                "params": [p for n, p in vae.named_parameters() if "conv_in" not in n],
                "lr": learning_rate_vae / vae_ch,
            },
            {
                "params": [p for n, p in vae.named_parameters() if "conv_in" in n],
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-3,
        betas=(0.9, 0.95),
    )

    optimizer_D = optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate_disc,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
    )

    lpips = LPIPS().cuda()

    dataloader = create_dataloader(
        dataset_url, batch_size, num_workers=1, do_shuffle=True
    )
    test_dataloader = create_dataloader(
        test_dataset_url, batch_size, num_workers=1, do_shuffle=False, just_resize=True
    )

    num_training_steps = max_steps
    num_warmup_steps = 200
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer_G, num_warmup_steps, num_training_steps
    )

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    global_step = 0

    if load_path is not None:
        state_dict = torch.load(load_path, map_location="cpu")
        try:
            status = vae.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(e)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            status = vae.load_state_dict(state_dict, strict=True)
            print(status)

    t0 = time.time()

    # lecam variable

    lecam_loss_weight = 0.1
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9

    for epoch in range(num_epochs):
        for i, real_images_hr in enumerate(dataloader):
            time_taken_till_load = time.time() - t0

            t0 = time.time()
            # resize real image to 256
            real_images_hr = real_images_hr[0].to(device)
            real_images_for_enc = F.interpolate(
                real_images_hr, size=(256, 256), mode="area"
            )
            if random.random() < 0.5:
                real_images_for_enc = torch.flip(real_images_for_enc, [-1])
                real_images_hr = torch.flip(real_images_hr, [-1])

            z = vae.encoder(real_images_for_enc)

            # z distribution
            with ctx:
                z_dist_value: torch.Tensor = z.detach().cpu().reshape(-1)

            def kurtosis(x):
                return ((x - x.mean()) ** 4).mean() / (x.std() ** 4)

            def skew(x):
                return ((x - x.mean()) ** 3).mean() / (x.std() ** 3)

            z_quantiles = {
                "0.0": z_dist_value.quantile(0.0),
                "0.2": z_dist_value.quantile(0.2),
                "0.4": z_dist_value.quantile(0.4),
                "0.6": z_dist_value.quantile(0.6),
                "0.8": z_dist_value.quantile(0.8),
                "1.0": z_dist_value.quantile(1.0),
                "kurtosis": kurtosis(z_dist_value),
                "skewness": skew(z_dist_value),
            }

            if do_clamp:
                z = z.clamp(-clamp_th, clamp_th)
            z_s = vae.reg(z)

            #### do aug

            if random.random() < 0.5 and flip_invariance:
                z_s = torch.flip(z_s, [-1])
                z_s[:, -4:-2] = -z_s[:, -4:-2]
                real_images_hr = torch.flip(real_images_hr, [-1])

            if random.random() < 0.5 and flip_invariance:
                z_s = torch.flip(z_s, [-2])
                z_s[:, -2:] = -z_s[:, -2:]
                real_images_hr = torch.flip(real_images_hr, [-2])

            if random.random() < 0.5 and crop_invariance:
                # crop image and latent.'

                # new_z_h, new_z_w, offset_z_h, offset_z_w
                z_h, z_w = z.shape[-2:]
                new_z_h = random.randint(12, z_h - 1)
                new_z_w = random.randint(12, z_w - 1)
                offset_z_h = random.randint(0, z_h - new_z_h - 1)
                offset_z_w = random.randint(0, z_w - new_z_w - 1)

                new_h = (
                    new_z_h * downscale_factor * 2
                    if decoder_also_perform_hr
                    else new_z_h * downscale_factor
                )
                new_w = (
                    new_z_w * downscale_factor * 2
                    if decoder_also_perform_hr
                    else new_z_w * downscale_factor
                )
                offset_h = (
                    offset_z_h * downscale_factor * 2
                    if decoder_also_perform_hr
                    else offset_z_h * downscale_factor
                )
                offset_w = (
                    offset_z_w * downscale_factor * 2
                    if decoder_also_perform_hr
                    else offset_z_w * downscale_factor
                )

                real_images_hr = real_images_hr[
                    :, :, offset_h : offset_h + new_h, offset_w : offset_w + new_w
                ]
                z_s = z_s[
                    :,
                    :,
                    offset_z_h : offset_z_h + new_z_h,
                    offset_z_w : offset_z_w + new_z_w,
                ]

                assert real_images_hr.shape[-2] == new_h
                assert real_images_hr.shape[-1] == new_w
                assert z_s.shape[-2] == new_z_h
                assert z_s.shape[-1] == new_z_w

            with ctx:
                reconstructed = vae.decoder(z_s)

            if global_step >= max_steps:
                break

            if do_ganloss:
                real_preds = discriminator(real_images_hr)
                fake_preds = discriminator(reconstructed.detach())
                d_loss, avg_real_logits, avg_fake_logits, disc_acc = gan_disc_loss(
                    real_preds, fake_preds, disc_type
                )

                lecam_anchor_real_logits = (
                    lecam_beta * lecam_anchor_real_logits
                    + (1 - lecam_beta) * avg_real_logits
                )
                lecam_anchor_fake_logits = (
                    lecam_beta * lecam_anchor_fake_logits
                    + (1 - lecam_beta) * avg_fake_logits
                )
                total_d_loss = d_loss.mean()
                d_loss_item = total_d_loss.item()
                if use_lecam:
                    # penalize the real logits to fake and fake logits to real.
                    lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(
                        2
                    ).mean() + (fake_preds - lecam_anchor_real_logits).pow(2).mean()
                    lecam_loss_item = lecam_loss.item()
                    total_d_loss = total_d_loss + lecam_loss * lecam_loss_weight

                optimizer_D.zero_grad()
                total_d_loss.backward(retain_graph=True)
                optimizer_D.step()

            # unnormalize the images, and perceptual loss
            _recon_for_perceptual = reconstructed

            if augment_before_perceptual_loss:
                real_images_hr_aug = real_images_hr.clone()
                if random.random() < 0.5:
                    _recon_for_perceptual = torch.flip(_recon_for_perceptual, [-1])
                    real_images_hr_aug = torch.flip(real_images_hr_aug, [-1])
                if random.random() < 0.5:
                    _recon_for_perceptual = torch.flip(_recon_for_perceptual, [-2])
                    real_images_hr_aug = torch.flip(real_images_hr_aug, [-2])

            else:
                real_images_hr_aug = real_images_hr

            percep_rec_loss = lpips(_recon_for_perceptual, real_images_hr_aug).mean()

            # mse, vae loss.
            recon_for_mse = reconstructed
            vae_loss, loss_data = vae_loss_function(
                real_images_hr, recon_for_mse, z
            )
            # gan loss
            if do_ganloss and global_step >= 0:
                recon_for_gan = reconstructed
                fake_preds = discriminator(recon_for_gan)
                real_preds_const = real_preds.clone().detach()
                if disc_type == "bce":
                    g_gan_loss = nn.functional.binary_cross_entropy_with_logits(
                        fake_preds, torch.ones_like(fake_preds)
                    )
                elif disc_type == "hinge":
                    g_gan_loss = -fake_preds.mean()

                overall_vae_loss = percep_rec_loss + g_gan_loss + vae_loss
                g_gan_loss = g_gan_loss.item()
            else:
                overall_vae_loss = percep_rec_loss + vae_loss
                g_gan_loss = 0.0

            overall_vae_loss.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()
            lr_scheduler.step()

            if do_ganloss:
                optimizer_D.zero_grad()

            time_taken_till_step = time.time() - t0

            
            if global_step % 5 == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": i,
                        "overall_vae_loss": overall_vae_loss.item(),
                        "mse_loss": loss_data["recon_loss"],
                        "kl_loss": loss_data["kl_loss"],
                        "perceptual_loss": percep_rec_loss.item(),
                        "gan/generator_gan_loss": (
                            g_gan_loss if do_ganloss else None
                        ),
                        "z_quantiles/abs_z": loss_data["average_of_abs_z"],
                        "z_quantiles/std_z": loss_data["std_of_abs_z"],
                        "z_quantiles/logvar": loss_data["average_of_logvar"],
                        "gan/avg_real_logits": (
                            avg_real_logits if do_ganloss else None
                        ),
                        "gan/avg_fake_logits": (
                            avg_fake_logits if do_ganloss else None
                        ),
                        "gan/discriminator_loss": (
                            d_loss_item if do_ganloss else None
                        ),
                        "gan/discriminator_accuracy": (
                            disc_acc if do_ganloss else None
                        ),
                        "gan/lecam_loss": lecam_loss_item if do_ganloss else None,
                        "gan/lecam_anchor_real_logits": (
                            lecam_anchor_real_logits if do_ganloss else None
                        ),
                        "gan/lecam_anchor_fake_logits": (
                            lecam_anchor_fake_logits if do_ganloss else None
                        ),
                        "z_quantiles/qs": z_quantiles,
                        "time_taken_till_step": time_taken_till_step,
                        "time_taken_till_load": time_taken_till_load,
                    }
                )

            if global_step % 200 == 0:

                wandb.log(
                    {
                        f"loss_stepwise/mse_loss_{global_step}": loss_data[
                            "recon_loss"
                        ],
                        f"loss_stepwise/kl_loss_{global_step}": loss_data[
                            "kl_loss"
                        ],
                        f"loss_stepwise/overall_vae_loss_{global_step}": overall_vae_loss.item(),
                    }
                )

            log_message = f"Epoch [{epoch}/{num_epochs}] - "
            log_items = [
                ("perceptual_loss", percep_rec_loss.item()),
                ("mse_loss", loss_data["recon_loss"]),
                ("kl_loss", loss_data["kl_loss"]),
                ("overall_vae_loss", overall_vae_loss.item()),
                ("ABS mu (0.0): average_of_abs_z", loss_data["average_of_abs_z"]),
                ("STD mu : std_of_abs_z", loss_data["std_of_abs_z"]),
                (
                    "ABS logvar (0.0) : average_of_logvar",
                    loss_data["average_of_logvar"],
                ),
                ("STD logvar : std_of_logvar", loss_data["std_of_logvar"]),
                *[(f"z_quantiles/{q}", v) for q, v in z_quantiles.items()],
                ("time_taken_till_step", time_taken_till_step),
                ("time_taken_till_load", time_taken_till_load),
            ]

            if do_ganloss:
                log_items = [
                    ("d_loss", d_loss_item),
                    ("gan_loss", g_gan_loss),
                    ("avg_real_logits", avg_real_logits),
                    ("avg_fake_logits", avg_fake_logits),
                    ("discriminator_accuracy", disc_acc),
                    ("lecam_loss", lecam_loss_item),
                    ("lecam_anchor_real_logits", lecam_anchor_real_logits),
                    ("lecam_anchor_fake_logits", lecam_anchor_fake_logits),
                ] + log_items

            log_message += "\n\t".join(
                [f"{key}: {value:.4f}" for key, value in log_items]
            )
            logger.info(log_message)

            global_step += 1
            t0 = time.time()

            if (
                evaluate_every_n_steps > 0
                and global_step % evaluate_every_n_steps == 1
            ):

                with torch.no_grad():
                    all_test_images = []
                    all_reconstructed_test = []

                    for test_images in test_dataloader:
                        test_images_ori = test_images[0].to(device)
                        # resize to 256
                        test_images = F.interpolate(
                            test_images_ori, size=(256, 256), mode="area"
                        )
                        with ctx:
                            z = vae.encoder(test_images)

                        if do_clamp:
                            z = z.clamp(-clamp_th, clamp_th)

                        z_s = vae.reg(z)

                        if flip_invariance:
                            z_s = torch.flip(z_s, [-1, -2])
                            z_s[:, -4:] = -z_s[:, -4:]

                        with ctx:
                            reconstructed_test = vae.decoder(z_s)

                        # unnormalize the images
                        test_images_ori = test_images_ori * 0.5 + 0.5
                        reconstructed_test = reconstructed_test * 0.5 + 0.5
                        # clamp
                        test_images_ori = test_images_ori.clamp(0, 1)
                        reconstructed_test = reconstructed_test.clamp(0, 1)

                        # flip twice
                        if flip_invariance:
                            reconstructed_test = torch.flip(
                                reconstructed_test, [-1, -2]
                            )

                        all_test_images.append(test_images_ori)
                        all_reconstructed_test.append(reconstructed_test)

                        if len(all_test_images) >= 2:
                            break

                    test_images = torch.cat(all_test_images, dim=0)
                    reconstructed_test = torch.cat(all_reconstructed_test, dim=0)

                    logger.info(f"Epoch [{epoch}/{num_epochs}] - Logging test images")

                    # crop test and recon to 64 x 64
                    D = 512 if decoder_also_perform_hr else 256
                    offset = 0
                    test_images = test_images[
                        :, :, offset : offset + D, offset : offset + D
                    ].cpu()
                    reconstructed_test = reconstructed_test[
                        :, :, offset : offset + D, offset : offset + D
                    ].cpu()

                    # concat the images into one large image.
                    # make size of (D * 4) x (D * 4)
                    recon_all_image = torch.zeros((3, D * 4, D * 4))
                    test_all_image = torch.zeros((3, D * 4, D * 4))

                    wandb.log(
                        {
                            "reconstructed_test_images": [
                                wandb.Image(recon_all_image),
                            ],
                            "test_images": [
                                wandb.Image(test_all_image),
                            ],
                        }
                    )

                    os.makedirs(f"./ckpt/{run_name}", exist_ok=True)
                    torch.save(
                        vae.state_dict(),
                        f"./ckpt/{run_name}/vae_epoch_{epoch}_step_{global_step}.pt",
                    )
                    print(
                        f"Saved checkpoint to ./ckpt/{run_name}/vae_epoch_{epoch}_step_{global_step}.pt"
                    )

    cleanup()


if __name__ == "__main__":
    train()