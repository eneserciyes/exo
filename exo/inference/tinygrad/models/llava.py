from typing import Optional

from .llama import Transformer
from tinygrad import Tensor
from tinygrad import nn

class VisionEmbeddings:
    def __init__(self, hidden_size: int, image_size: int, patch_size: int, num_channels: int):
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = Tensor.zeros((self.embed_dim,))

        self.patch_embedding = nn.Conv2d(num_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.positional_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        patch_embeddings = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1) # (B, embed_dim, H, W) -> (B, num_patches, embed_dim)
        cls_embeddings = self.class_embedding.reshape(1, 1, -1).repeat(B, 1, 1) # (embed_dim,) -> (1, 1, embed_dim) -> (B, 1, embed_dim)
        embeddings = Tensor.cat([cls_embeddings, patch_embeddings], axis=1) # (B, num_patches + 1, embed_dim)
        embeddings += self.positional_embedding.weight # (B, num_patches + 1, embed_dim)
        return embeddings



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
        self.language_model = Transformer() # TODO: params