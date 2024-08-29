import torch
import torch.nn as nn

# CNN
class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1,16,kernel_size=(3,3))
    self.conv2 = nn.Conv2d(16,32,kernel_size=(3,3))
    self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
    self.lin1 = nn.Linear(800,128)
    self.out = nn.Linear(128,10)
  def forward(self,x):
    x = self.conv1(x)
    x = nn.functional.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = nn.functional.relu(x)
    x = self.maxpool(x)
    x = x.flatten(start_dim=1)
    x = self.lin1(x) 
    x = nn.functional.relu(x)
    x = self.out(x)
    x = nn.functional.log_softmax(x,dim=1)
    return x

# CNNWithAttention
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key(x).view(batch_size, -1, width * height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = torch.softmax(energy, dim=-1)  # B x N x N
        proj_value = self.value(x).view(batch_size, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x  # 加上殘差
        return out

class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.attention = SelfAttention(in_dim=32)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.lin1 = nn.Linear(800, 128)
        self.out = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        
        x = self.attention(x)
        
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

# Transformer
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=7, emb_size=128, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, emb_size, num_patches) -> (B, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, emb_size) -> (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, emb_size)
        x = x + self.pos_embed
        return x

class Transformer(nn.Module):
    def __init__(self, emb_size=128, num_heads=4, num_layers=2, num_classes=10, mlp_dim=256):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size=emb_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=num_layers
        )
        self.cls_head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        cls_token = x[:, 0]  # 使用CLS token作為分類的特徵
        x = self.cls_head(cls_token)
        return nn.functional.log_softmax(x, dim=1)