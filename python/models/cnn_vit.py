import torch
import torch.nn as nn
import math
from models.simple_vit import TransformerEncoder


class ImagePatcher(nn.Module):
    def __init__(self, in_size=1, out_size=4, patch_size=4):
        super(ImagePatcher, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.patch_size = patch_size
        self.cnn_block = torch.nn.Conv2d(
            in_size, self.out_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.cnn_block(x)
        y = y.reshape(b, self.out_size, -1)
        y = y.transpose(1, 2).contiguous()  # (batch, num_patches, features)
        return y


class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_size=128,
        num_encoder_blocks=3,
        num_classes=10,
    ):
        super(ViT, self).__init__()
        height, width, channels = img_size
        assert height == width  # for now, just square images
        self.num_patches = int(height / patch_size) ** 2
        self.patch_dim = patch_size**2 * channels
        self.patch_size = patch_size

        self.embed_size = embed_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_classes = num_classes

        self.image_patcher = ImagePatcher(
            in_size=channels, out_size=embed_size, patch_size=patch_size
        )
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches + 1, embed_size), requires_grad=True
        )
        self.class_token = nn.Parameter(
            data=torch.randn(1, 1, embed_size), requires_grad=True
        )

        self.encoders = torch.nn.ModuleList(
            [
                TransformerEncoder(embed_size=embed_size, num_patches=self.num_patches)
                for _ in range(num_encoder_blocks)
            ]
        )
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.image_patcher(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_token, x), 1) + self.positional_embeddings
        z = embeddings
        for encoder in self.encoders:
            z = encoder(z)

        cls = self.classifier(z[:, 0])
        return cls


if __name__ == "__main__":
    model = ViT()
    x = torch.rand(2, 3, 128, 128)
    print(x.shape)
    y = model(x)
    print(y.shape)
