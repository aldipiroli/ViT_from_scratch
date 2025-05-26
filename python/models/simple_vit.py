import torch
import torch.nn as nn
import math
from utils import divide_in_patches

class SelfAttention(nn.Module):
    def __init__(self, in_size=128, out_size=128):
        super(SelfAttention, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.K = nn.Linear(self.in_size, self.out_size)
        self.Q = nn.Linear(self.in_size, self.out_size)
        self.V = nn.Linear(self.in_size, self.out_size)

    def forward(self, x):
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)

        qk = torch.matmul(q, k.transpose(-2, -1))
        attention = nn.functional.softmax(qk / math.sqrt(self.out_size), -1)
        self_attention = torch.matmul(attention, v)
        return self_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_size=128, out_size=128, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.attention_heads = torch.nn.ModuleList(
            [
                SelfAttention(self.in_size, int(self.out_size / 3))
                for i in range(self.num_heads)
            ]
        )
        self.projection_matrix = nn.Linear(
            int(self.out_size / 3) * num_heads, self.out_size
        )

    def forward(self, x):
        all_self_attentions = []
        for attention_head in self.attention_heads:
            z = attention_head(x)
            all_self_attentions.append(z)
        all_self_attentions = torch.cat(all_self_attentions, -1)
        z = self.projection_matrix(all_self_attentions)
        return z


class TransformerEncoder(nn.Module):

    def __init__(self, embed_size=128, num_patches=16):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.num_patches = num_patches

        self.ln = nn.LayerNorm(embed_size)
        self.multi_head_self_attention = MultiHeadSelfAttention(
            in_size=self.embed_size, out_size=self.embed_size, num_heads=3
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size * 2),
            nn.GELU(),
            nn.Linear(self.embed_size * 2, self.embed_size),
        )

    def forward(self, z):
        z0 = self.multi_head_self_attention(self.ln(z))
        z1 = z + z0
        z = self.mlp(self.ln(z1))
        z = z + z1
        return z


class SimpleViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_size=128,
        num_encoder_blocks=3,
        num_classes=10,
    ):
        super(SimpleViT, self).__init__()
        height, width, channels = img_size
        assert height == width  # for now, just square images
        self.num_patches = int(height / patch_size) ** 2
        self.patch_dim = patch_size**2 * channels
        self.patch_size = patch_size

        self.embed_size = embed_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_classes = num_classes

        self.patch_embed_transform = nn.Linear(self.patch_dim, self.embed_size)
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches + 1, embed_size), requires_grad=True
        )
        self.class_token = nn.Parameter(
            data=torch.randn(1, 1, embed_size), requires_grad=True
        )

        self.encoders = torch.nn.ModuleList(
            [
                TransformerEncoder(embed_size=embed_size, num_patches=self.num_patches)
                for i in range(num_encoder_blocks)
            ]
        )
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        embeddings = (
            torch.cat((class_token, self.patch_embed_transform(x)), 1)
            + self.positional_embeddings
        )
        z = embeddings
        for encoder in self.encoders:
            z = encoder(z)

        cls = self.classifier(z[:, 0])
        return cls


if __name__ == "__main__":
    model = ViT()
    x = torch.rand(2, 128, 128, 3)
    print(x.shape)
    x = divide_in_patches(x, patch_size=32)
    y = model(x)
    print(y.shape)
