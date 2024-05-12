import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 64

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            noise_feature = self.noise_func(noise_embed).view(batch, -1, 1, 1)
            x = x + noise_feature
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer_fn(nn.Module):
    def __init__(self,
                 in_channel=3,
                 out_channel=3,
                 dim=36,
                 num_blocks=[2, 2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):

        super(Restormer_fn, self).__init__()

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

        self.patch_embed = OverlapPatchEmbed(in_channel, dim)
        self.patch_embed_refine = OverlapPatchEmbed(in_channel * 2, dim)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                 LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level1 = nn.Sequential(*layers)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                 LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level1_refine = nn.Sequential(*layers)

        self.reduce_chan_level1_refine = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level2 = nn.Sequential(*layers)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level2_refine = nn.Sequential(*layers)

        self.reduce_chan_level2_refine = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.decoder_level2 = nn.Sequential(*layers)

        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.decoder_level1 = nn.Sequential(*layers)

        self.reduce_chan_level_out = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, vis_ir_img, inp_img, time):
        t = self.noise_level_mlp(time)
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_refine = self.patch_embed_refine(vis_ir_img)

        for layer in self.encoder_level1:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level1 = layer(inp_enc_level1, t)
            else:
                inp_enc_level1 = layer(inp_enc_level1)
        out_enc_level1 = inp_enc_level1

        for layer in self.encoder_level1_refine:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level1_refine = layer(inp_enc_level1_refine, t)
            else:
                inp_enc_level1_refine = layer(inp_enc_level1_refine)
        out_enc_level1_refine = inp_enc_level1_refine

        out_enc_level1 = torch.cat([out_enc_level1, out_enc_level1_refine], 1)
        out_enc_level1 = self.reduce_chan_level1_refine(out_enc_level1)

        inp_enc_level2 = out_enc_level1
        for layer in self.encoder_level2:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level2 = layer(inp_enc_level2, t)
            else:
                inp_enc_level2 = layer(inp_enc_level2)
        out_enc_level2 = inp_enc_level2

        inp_enc_level2_refine = out_enc_level1_refine
        for layer in self.encoder_level2_refine:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level2_refine = layer(inp_enc_level2_refine, t)
            else:
                inp_enc_level2_refine = layer(inp_enc_level2_refine)
        out_enc_level2_refine = inp_enc_level2_refine

        out_enc_level2 = torch.cat([out_enc_level2, out_enc_level2_refine], 1)
        out_enc_level2 = self.reduce_chan_level2_refine(out_enc_level2)

        inp_dec_level2 = out_enc_level2
        for layer in self.decoder_level2:
            if isinstance(layer, FeatureWiseAffine):
                inp_dec_level2 = layer(inp_dec_level2, t)
            else:
                inp_dec_level2 = layer(inp_dec_level2)
        out_dec_level2 = inp_dec_level2

        inp_dec_level1 = torch.cat([out_dec_level2, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)

        for layer in self.decoder_level1:
            if isinstance(layer, FeatureWiseAffine):
                inp_dec_level1 = layer(inp_dec_level1, t)
            else:
                inp_dec_level1 = layer(inp_dec_level1)
        out_dec_level1 = inp_dec_level1

        out_dec_level1 = torch.cat([out_dec_level1, inp_enc_level1], 1)
        out_dec_level1 = self.reduce_chan_level_out(out_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1