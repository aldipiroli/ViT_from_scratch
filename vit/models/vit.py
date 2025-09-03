import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ImagePatcher(nn.Module):
    def __init__(self, out_size=1, patch_size=4):
        super(ImagePatcher, self).__init__()
        self.out_size = out_size
        self.patch_size = patch_size

    def forward(self, x):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        x_p = unfold(x)  # B, D, N
        x_p = x_p.permute(0, 2, 1)
        return x_p


class ImagePatcherCNN(nn.Module):
    def __init__(self, in_size=1, out_size=4, patch_size=4):
        super(ImagePatcherCNN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.patch_size = patch_size
        self.cnn_block = torch.nn.Conv2d(in_size, self.out_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        b = x.shape[0]
        y = self.cnn_block(x)
        y = y.reshape(b, self.out_size, -1)
        y = y.transpose(1, 2).contiguous()  # (batch, num_patches, features)
        return y


class SelfAttention(nn.Module):
    def __init__(self, embed_size=128, n_heads=8):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.inter_embed_size = embed_size // n_heads

        self.Ukqv = nn.Linear(self.embed_size, 3 * self.inter_embed_size)

    def forward(self, z):
        b,n = z.shape[0],z.shape[1]
        kqv = self.Ukqv(z)
        kqv = kqv.reshape(b,n,self.inter_embed_size,3)
        k = kqv[..., 0]
        q = kqv[..., 1]
        v = kqv[..., 2]

        qk = torch.matmul(q, k.transpose(-2, -1))
        attention = nn.functional.softmax(qk / math.sqrt(self.inter_embed_size), -1)
        self_attention = torch.matmul(attention, v)
        return self_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size=128, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_heads = torch.nn.ModuleList(
            [SelfAttention(self.embed_size, self.num_heads) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear((self.embed_size // num_heads) * num_heads, self.embed_size)

    def forward(self, z):
        all_self_attentions = []
        for attention_head in self.attention_heads:
            z_msa = attention_head(z)
            all_self_attentions.append(z_msa)
        all_self_attentions = torch.cat(all_self_attentions, -1)
        z = self.projection_matrix(all_self_attentions)
        return z

class ParallelMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size=128, num_heads=8):
        super(ParallelMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.intermediate_embed_size = embed_size // num_heads
        self.kqv_projection = nn.Linear(embed_size, 3* num_heads * self.intermediate_embed_size)

    def forward(self, z):
        b, n = z.shape[0],z.shape[1]
        kvq = self.kqv_projection(z)
        kvq = kvq.reshape(b, n, self.num_heads, self.intermediate_embed_size, 3) # (batch, n_patches, n_heads, embed_size, 3)
        kvq = kvq.permute(2, 0, 1, 3, 4)
        k = kvq[..., 0] # (n_heads, batch, n_patches, embed_size)
        v = kvq[..., 1]
        q = kvq[..., 2]

        qk = q @ k.transpose(3,2)
        a = F.softmax(qk, -1) * self.intermediate_embed_size**-0.5
        sa = a @ v # (n_heads, batch, n_patches, embed_size)
        mhsa = sa.permute(1,2,0,3).reshape(b, n, self.num_heads*self.intermediate_embed_size) # aggregate heads
        return mhsa


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size=128, num_patches=16, num_heads=8, use_parallel=False):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.num_patches = num_patches

        self.ln = nn.LayerNorm(embed_size)
        if not use_parallel: 
            self.multi_head_self_attention = MultiHeadSelfAttention(
                embed_size=self.embed_size,  num_heads=num_heads
            )
        else:
            self.multi_head_self_attention = ParallelMultiHeadSelfAttention(
                embed_size=self.embed_size, num_heads=num_heads
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
    def __init__(self, config):
        super(SimpleViT, self).__init__()
        self.config = config
        height, width, channels = config["MODEL"]["img_size"]
        assert height == width  # for now, just square images
        self.patch_size = config["MODEL"]["patch_size"]

        self.num_patches = (height // self.patch_size) ** 2
        self.patch_dim = self.patch_size**2 * channels

        self.embed_size = config["MODEL"]["embed_size"]
        self.num_encoder_blocks = config["MODEL"]["num_encoder_blocks"]
        self.num_classes = config["MODEL"]["num_classes"]
        self.num_mha_heads = config["MODEL"]["num_mha_heads"]

        self.patch_embed_transform = nn.Linear(self.patch_dim, self.embed_size)
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches + 1, self.embed_size), requires_grad=True
        )
        self.class_token = nn.Parameter(data=torch.randn(1, 1, self.embed_size), requires_grad=True)

        self.encoders = torch.nn.ModuleList(
            [
                TransformerEncoder(embed_size=self.embed_size, num_patches=self.num_patches, num_heads=self.num_mha_heads)
                for _ in range(self.num_encoder_blocks)
            ]
        )
        self.classifier = nn.Linear(self.embed_size, self.num_classes)

        if config["MODEL"]["patch_type"] == "simple":
            self.image_patcher = ImagePatcher(out_size=self.embed_size, patch_size=self.patch_size)
        else:
            self.image_patcher = ImagePatcherCNN(out_size=self.patch_dim, patch_size=self.patch_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.image_patcher(x)

        class_token = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_token, self.patch_embed_transform(x)), 1)
        embeddings += self.positional_embeddings

        z = embeddings
        for encoder in self.encoders:
            z = encoder(z)

        cls = self.classifier(z[:, 0])
        return cls
