import torch
from torch import nn
import torchvision as TV
import torchvision.transforms as transforms
import model

# 定義轉換，包括轉為Tensor和正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和標準差
])

test_data = TV.datasets.MNIST("MNIST/", train=False, transform=transform, download=True)

def test_acc(M):
    M.eval()  # 設置模型為評估模式
    acc = 0.
    xt = test_data.data.unsqueeze(1).float() / 255  # 將數據轉換為float並縮放到[0, 1]範圍
    xt = (xt - 0.1307) / 0.3081  # 使用MNIST的均值和標準差進行正規化
    yt = test_data.targets.detach()
    
    with torch.no_grad():  # 禁用梯度計算，節省內存和加速計算
        preds = M(xt)
        pred_ind = torch.argmax(preds, dim=1)
        acc = (pred_ind == yt).sum().float() / len(test_data)
    
    return acc, xt, yt

# M = model.CNN()
M = model.CNNWithAttention()
# M = model.TransformerClassifier()

M.load_state_dict(torch.load("mnist.pt", weights_only=True))

acc2, xt2, yt2 = test_acc(M)
print('Testing accuracy =', (acc2 * 100).item(), '%')
