from typing import Optional, Dict, Union
from dataclasses import dataclass
import inspect

from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv

from exo.inference.shard import Shard

@dataclass
class VisionConfig:
  model_type: str
  num_hidden_layers: int = 24
  hidden_size: int = 1024
  intermediate_size: int = 4096
  num_attention_heads: int = 16
  image_size: int = 336
  patch_size: int = 14
  projection_dim: int = 768
  vocab_size: int = 32000
  num_channels: int = 3
  layer_norm_eps: float = 1e-5

  @classmethod
  def from_dict(cls, params):
    return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

class VisionEmbeddings:
    def __init__(self, config: VisionConfig):
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Tensor.zeros((self.embed_dim,))

        self.patch_embedding = nn.Conv2d(config.num_channels, self.embed_dim, kernel_size=config.patch_size, stride=config.patch_size, bias=False)

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.positional_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        patch_embeddings = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1) # (B, embed_dim, H, W) -> (B, num_patches, embed_dim)
        cls_embeddings = self.class_embedding.reshape(1, 1, -1).repeat(B, 1, 1) # (embed_dim,) -> (1, 1, embed_dim) -> (B, 1, embed_dim)
        embeddings = Tensor.cat([cls_embeddings, patch_embeddings], axis=1) # (B, num_patches + 1, embed_dim)
        embeddings += self.positional_embedding.weight # (B, num_patches + 1, embed_dim)
        return embeddings

class VisionAttention:
    def __init__(self, dim: int, n_heads: int):
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, mask=None):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        bsz, seqlen, _, _ = xq.shape
        attn = xq.scaled_dot_product_attention(xk, xv, mask).transpose(1, 2)
        attn = attn.reshape(bsz, seqlen, -1)
        return self.proj(attn)


class VisionMLP:
    def __init__(self, config: VisionConfig):
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).quick_gelu())

class VisionEncoderLayer:
    def __init__(self, config: VisionConfig):
        self.embed_dim = config.hidden_size
        self.self_attn = VisionAttention(config.hidden_size, config.num_attention_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        y = self.layer_norm2(x := x + y)
        y = self.mlp(y)
        return x + y

class VisionEncoder:
    def __init__(self, config: VisionConfig):
        self.layers = [VisionEncoderLayer(config) for _ in config.num_hidden_layers]


class ClipVisionModel:
    def __init__(self, hidden_size):
        self.embeddings = VisionEmbeddings()
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.encoder = VisionEncoder()
        self.post_ln = nn.LayerNorm(hidden_size)

    def __call__(self, x: Tensor, output_hidden_states: Optional[bool]=False):
        x = self.embeddings(x)
        x = self.pre_ln(x)

        encoder_states = (x,) if output_hidden_states else None
        for l in self.encoder.layers:
            x = l(x)
            if output_hidden_states:
                encoder_states += (x,)

        out = self.post_ln(x[:, 0, :]) # TODO: check what this does
        return out, x, encoder_states

class LlavaMultimodalProjector:
    pass

class Llava:
    def __init__(self):
        self.vision_tower = ClipVisionModel()
        self.multimodal_projector = LlavaMultimodalProjector()