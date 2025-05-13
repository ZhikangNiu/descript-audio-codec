import math
from typing import List
from typing import Union
import logging
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.model.utils import make_pad_mask
from .bigvgan import BigVGAN
from .vocos import VocosDecoder
from .regulator import InterpolateRegulator
from .attn_proj import AttnProjection
import json
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=0 if stride % 2 == 0 else 1
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResidualBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # 替换为 LayerNorm
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)   # 替换为 LayerNorm
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        vae_dim: Union[int, list] = 8,
        sample_rate: int = 44100,
        distill: bool = False,
        distill_hidden_dim: int = 1024,
        decoder_type : str = "dac", # bigvgan | dac
        pre_vae_block: bool = False,
        attn_proj: bool = False,
        post_vae_block: bool = False,
        bigvgan_conf: str = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/conf/bigvgan_conf/bigvgan_v2_24khz_100band_256x.json",
        align_space="ssl", # ssl means up vae feature -> ssl feature | vae means ssl feature -> vae feature
        sampling_ratios=[0,1],
        distill_loss_dim = -1,
        masked_mean = True,
    ):
        super().__init__()
        logger.info(
            f"encoder_dim: {encoder_dim} | encoder rates: {encoder_rates} | latent_dim: {latent_dim}",
            f"decoder_dim: {decoder_dim} | decoder rates: {decoder_rates} | decoder type :{decoder_type} | vae_dim: {vae_dim}",
            f"sample rate: {sample_rate} | distill: {distill} | distill_hidden_dim:{distill_hidden_dim}",
            f"pre_vae_block: {pre_vae_block} | attn_proj: {attn_proj} | post_vae_block:{post_vae_block}",
            f"bigvgan conf: {bigvgan_conf}"
            f"align_space: {align_space} | sampling_ratios: {sampling_ratios} | distill_loss_dim: {distill_loss_dim} -1 means T avg, 0 means dim"
        )
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.sample_rate = sample_rate
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.vae_dim = vae_dim
        self.attn_proj = attn_proj
        self.pre_vae_block = pre_vae_block
        self.post_vae_block = post_vae_block
        
        if self.pre_vae_block:
            self.pre_block = self._build_residual_blocks(latent_dim,self.vae_dim)
        elif self.attn_proj:
            # in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2
            self.pre_block = AttnProjection(latent_dim,self.vae_dim,num_heads=8)
        else:
            self.pre_block = nn.Linear(latent_dim,self.vae_dim)
        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_var = nn.Linear(self.vae_dim, self.vae_dim)
        
        if self.post_vae_block:
            self.decoder_proj = self._build_residual_blocks(self.vae_dim,latent_dim)
        elif self.attn_proj:
            self.decoder_proj = AttnProjection(self.vae_dim,latent_dim,8)
        else:
            self.decoder_proj = nn.Linear(self.vae_dim,latent_dim)
        
        self.decoder_type = decoder_type
        if self.decoder_type == "dac":
            self.decoder = Decoder(
                latent_dim,
                decoder_dim,
                decoder_rates,
            )
            self.apply(init_weights)
        elif self.decoder_type == "bigvgan":
            self.bigvgan_conf = bigvgan_conf
            with open(self.bigvgan_conf) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            self.decoder = BigVGAN(h)
        elif self.decoder_type == "vocos":
            self.bigvgan_conf = bigvgan_conf
            with open(self.bigvgan_conf) as f:
                data = f.read()
            json_config = json.loads(data) # dict
            self.decoder = VocosDecoder(**json_config)
        else:
            raise ValueError(f"Invalid decoder type: {decoder_type}")   
        self.distill = distill
        self.distill_hidden_dim = distill_hidden_dim
        self.sampling_ratios = sampling_ratios
        if align_space == "ssl":
            proj_dim = self.distill_hidden_dim * 2
            self.projectors = InterpolateRegulator(self.sampling_ratios, self.vae_dim,proj_dim,self.distill_hidden_dim)
        elif align_space == "vae":
            proj_dim = self.vae_dim * 2
            self.projectors = InterpolateRegulator(self.sampling_ratios, self.distill_hidden_dim, proj_dim, self.vae_dim)
        
        self.distill_loss_dim = distill_loss_dim
        self.masked_mean = masked_mean
        self.align_space = align_space
        self.delay = self.get_delay()

    def _build_residual_blocks(self, in_dim, out_dim):
        layers = []
        current_dim = in_dim
        max_depth = 3
        
        # 统一处理维度变换
        if in_dim != out_dim:
            step_fn = (lambda x: max(x//2, out_dim)) if in_dim > out_dim else (lambda x: min(x*2, out_dim))
            for _ in range(max_depth):
                next_dim = step_fn(current_dim)
                layers.append(ResidualBottleneck(current_dim, next_dim))
                current_dim = next_dim
                if current_dim == out_dim:
                    break
            # 强制最终维度对齐
            if current_dim != out_dim:
                layers.append(nn.Sequential(
                    nn.Linear(current_dim, out_dim),
                    # nn.BatchNorm1d(out_dim),
                    nn.GELU()
                ))
        else:
            # 维度相同则构建恒等残差块
            layers.append(ResidualBottleneck(in_dim, out_dim))
        
        return nn.Sequential(*layers)
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) :
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def compute_kl_loss(self,mu, log_var):
        # KL Loss 公式
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var.exp() + 1e-6), dim=-1)  # 按最后一维求和
        return kl_loss.mean()  # 求 batch 的平均值
    
    def encode(
        self,
        audio_data: torch.Tensor
    ):
        z = self.encoder(audio_data).transpose(1,2) # torch.Size([72, 1024, 29]),[B x D x T] -> torch.Size([72, 29, 1024]),[B x T x D] ->vq torch.Size([72, 29, 8]),[B x D x T]
        z = self.pre_block(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        log_var = torch.clamp(log_var, min=-12, max=12) # log var可能会爆掉
        
        # z_hat = self.decoder_proj(self.reparameterize(mu,log_var)).transpose(1,2)
        z_hat = self.reparameterize(mu,log_var)
        kl_loss = self.compute_kl_loss(mu,log_var)
        
        return z_hat, mu, log_var, kl_loss

    def decode(self, z: torch.Tensor):
        if self.decoder_type == "dac":
            z = self.decoder_proj(z).transpose(1,2) 
            recon = self.decoder(z)
        elif self.decoder_type == "bigvgan":
            recon = self.decoder(z.transpose(1,2))    # bigvgan 需要的输入是 B, T, D
        elif self.decoder_type == "vocos":
            recon = self.decoder(z.transpose(1,2)).unsqueeze(1)
        return recon

    def forward(
        self,
        audio_data: torch.Tensor, # B, 1, T (duration)
        sample_rate: int = None,
        guidance: torch.Tensor = None # B, T, D
    ):
        bsz, length = audio_data.shape[0], audio_data.shape[-1] # audio_data: B,1,T
        audio_data = self.preprocess(audio_data, sample_rate)
        z, mu, log_var, kl_loss = self.encode(audio_data) # z.shape = B, T, D
        proj_loss = 0.
        if self.distill and self.training:
            guidance_lengths = [g.shape[0] for g in guidance]
            z_lens = [zi.shape[0] for zi in z]
            target_lengths = torch.tensor(guidance_lengths, device=z.device)
            z_lengths = torch.tensor(z_lens, device = z.device)
            z_mask = make_pad_mask(z_lengths, max_len=torch.max(target_lengths)) # 16 150, padding的部分是1
            g_mask = make_pad_mask(target_lengths,max_len=torch.max(z_lengths))
            if self.align_space == "ssl":
                proj_z, olens = self.projectors(z, z_lengths, target_lengths)
                bsz, seq_len, distill_dim = proj_z.shape
                for i, (pi, gi) in enumerate(zip(proj_z,guidance)):
                    cos_sim = F.cosine_similarity(
                        pi, # 16, 150, 1024
                        gi, # 16, 150, 1024
                        dim = self.distill_loss_dim
                    )
                    if self.distill_loss_dim == -1:
                        if self.masked_mean:
                            proj_loss += masked_mean(-cos_sim, ~z_mask[i])
                        else:
                            proj_loss += -cos_sim.sum() / seq_len
                    elif self.distill_loss_dim == 0:
                        # print(cos_sim.shape)
                        proj_loss += -cos_sim.sum() / distill_dim
            elif self.align_space == "vae":
                proj_g, olens = self.projectors(guidance, target_lengths, z_lengths)
                # print(f"guidance shape {guidance.shape} | proj_g shape {proj_g.shape}")
                # print(g_mask)
                bsz, seq_len, distill_dim = proj_g.shape
                for i, (pi, gi) in enumerate(zip(z, proj_g)):
                    cos_sim = F.cosine_similarity(
                        pi, # 16, 150, 32
                        gi, # 16, 150, 32
                        dim = -1
                    )
                    if self.distill_loss_dim == -1:
                        if self.masked_mean:
                            proj_loss += masked_mean(-cos_sim, ~g_mask[i])
                        else:
                            proj_loss += -cos_sim.sum() / seq_len
                    elif self.distill_loss_dim == 0:
                        # print(cos_sim.shape)
                        proj_loss += -cos_sim.sum() / distill_dim
            proj_loss = proj_loss / bsz
            
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "vae/kl_loss": kl_loss,
            "vae/proj_loss": proj_loss
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)
