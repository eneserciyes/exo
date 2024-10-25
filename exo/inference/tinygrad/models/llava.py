from typing import Optional, Dict, Union
from dataclasses import dataclass
import inspect


from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv

from .llama import precompute_freqs_cis, apply_rotary_emb, repeat_kv, sample
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
    patch_embeddings = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1)  # (B, embed_dim, H, W) -> (B, num_patches, embed_dim)
    cls_embeddings = self.class_embedding.reshape(1, 1, -1).repeat(B, 1, 1)  # (embed_dim,) -> (1, 1, embed_dim) -> (B, 1, embed_dim)
    embeddings = Tensor.cat([cls_embeddings, patch_embeddings], axis=1)  # (B, num_patches + 1, embed_dim)
    embeddings += self.positional_embedding.weight  # (B, num_patches + 1, embed_dim)
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

  def __call__(self, x: Tensor, output_hidden_states: Optional[bool] = False):
    x = self.embeddings(x)
    x = self.pre_ln(x)

    encoder_states = (x,) if output_hidden_states else None
    for l in self.encoder.layers:
      x = l(x)
      if output_hidden_states:
        encoder_states += (x,)

    out = self.post_ln(x[:, 0, :])  # TODO: check what this does
    return out, x, encoder_states


class LlavaMultimodalProjector:
  pass


@dataclass
class TextConfig:
  model_type: str
  hidden_size: int = 4096
  num_hidden_layers: int = 32
  intermediate_size: int = 11008
  num_attention_heads: int = 32
  max_context: int = 1024 # TODO: check this
  head_dim: int = None
  rms_norm_eps: float = 1e-6
  vocab_size: int = 32000
  num_key_value_heads: int = None
  rope_theta: float = 10000
  rope_traditional: bool = False
  rope_scaling: Optional[Dict[str, Union[float, str]]] = None

  @classmethod
  def from_dict(cls, params):
    return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

  def __post_init__(self):
    if self.num_key_value_heads is None:
      self.num_key_value_heads = self.num_attention_heads

    if self.head_dim is None:
      self.head_dim = self.hidden_size // self.num_attention_heads

    if self.model_type is None:
      self.model_type = "llama"

    if self.rope_scaling:
      required_keys = {"factor", "type"}
      if not all(key in self.rope_scaling for key in required_keys):
        raise ValueError(f"rope_scaling must contain keys {required_keys}")

      if self.rope_scaling["type"] != "linear":
        raise ValueError("rope_scaling 'type' currently only supports 'linear'")


class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, max_context, linear=nn.Linear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.max_context = max_context

    self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor]) -> Tensor:
    if getenv("WQKV"):
      if not hasattr(self, "wqkv"):
        self.wqkv = Tensor.cat(self.wq.weight, self.wk.weight, self.wv.weight)
      xqkv = x @ self.wqkv.T
      xq, xk, xv = xqkv.split([self.wq.weight.shape[0], self.wk.weight.shape[0], self.wv.weight.shape[0]], dim=2)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    # create kv cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
      if isinstance(x.device, tuple):
        # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
        self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

    # update the cache
    assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"{xk.dtype=}, {xv.dtype=}, {self.cache_kv.dtype=}"
    self.cache_kv.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(Tensor.stack(xk, xv)).realize()

    keys = self.cache_kv[0].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xk
    values = self.cache_kv[1].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xv

    keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    return self.wo(attn)

class TextMLP:
  def __init__(self, dim: int, hidden_dim: int):
    self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).silu()*self.up_proj(x)) # SwiGLU


class TransformerBlock:
  def __init__(self, config: TextConfig):
    self.attention = Attention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.max_context)
    self.mlp = TextMLP(config.hidden_size, config.intermediate_size)
    self.attention_norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.ffn_norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor]):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    return (h + self.mlp(self.ffn_norm(h))).contiguous()


class Llama:
  def __init__(self, config: TextConfig, shard: Shard = None, jit: bool = True):
    self.config = config
    self.shard = shard
    self.vocab_size = config.vocab_size
    self.model_type = config.model_type
    self.num_hidden_layers = config.num_hidden_layers
    self.num_key_value_heads = config.num_key_value_heads
    self.head_dim = config.head_dim
    assert self.vocab_size > 0

    if self.shard.is_first_layer():
      self.tok_embeddings = nn.Embedding(self.vocab_size, config.hidden_size)
    self.layers = []
    for i in range(self.num_hidden_layers):
      if self.shard.start_layer <= i <= self.shard.end_layer:
        self.layers.append(TransformerBlock(config))
      else:
        self.layers.append(lambda x: x)  # TODO: check if this is correct, probably not

    if self.shard.is_last_layer():
      self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    # self.output = nn.Linear(dim, vocab_size, bias=False) # TODO: add this to LanguageModel
    self.max_context = config.max_context # TODO: add this to the config
    self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, self.max_context * 2, config.rope_theta).contiguous()
    self.forward_jit = TinyJit(self.forward) if jit else None
    self.shard = shard

  def forward(self, inputs: Tensor, start_pos: Union[Variable, int], inputs_embeds=None):
    seqlen = inputs.shape[1]
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))

    if inputs_embeds is None:
      if self.shard.is_first_layer():
        h = self.tok_embeddings(inputs)
      else:
        h = inputs
    else:
      h = inputs_embeds

    mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-100000000"), dtype=h.dtype, device=h.device).triu(start_pos + 1).realize() if seqlen > 1 else None

    for i in range(self.shard.start_layer, self.shard.end_layer + 1):
      layer = self.layers[i]
      h = layer(h, start_pos, freqs_cis, mask)

    if self.shard.is_last_layer():
      h = self.norm(h)
    return h

  def __call__(self, tokens: Tensor, start_pos: Variable):
    # TODO: better way to handle the first call v.s. the rest?
    if tokens.shape[0:2] == (1, 1) and self.forward_jit is not None:
      return self.forward_jit(tokens, Variable("start_pos", 0, self.max_context).bind(start_pos))
    return self.forward(tokens, start_pos)


class Llava:
  def __init__(self):
    self.vision_tower = ClipVisionModel()
    self.multimodal_projector = LlavaMultimodalProjector()
