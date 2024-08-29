import torch
from torch import nn
import torchvision as TV
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import model

# 設定資料轉換（包括轉為Tensor和正規化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和標準差
])

# 載入MNIST資料集
train_data = TV.datasets.MNIST("MNIST/", train=True, transform=transform, download=True)
test_data = TV.datasets.MNIST("MNIST/", train=False, transform=transform, download=True)

print(f'Number of samples in train_data: {len(train_data)}')
print(f'Number of samples in test_data: {len(test_data)}')

# 建立CNN模型
# M = model.CNN()
M = model.CNNWithAttention()
# M = model.TransformerClassifier()

# 訓練參數設定
epochs = 100
batch_size = 500
lr = 1e-3
opt = torch.optim.Adam(M.parameters(), lr=lr)
lossfn = nn.NLLLoss()

# 開始訓練
for epoch in range(epochs):
    # 隨機選取一個批次的數據
    batch_ids_CNN = np.random.randint(0, len(train_data), size=batch_size)
    xt = train_data.data[batch_ids_CNN].unsqueeze(1).float() / 255.0
    yt = train_data.targets[batch_ids_CNN]
    
    # 模型預測
    pred = M(xt)
    
    # 計算損失
    loss = lossfn(pred, yt)
    
    # 清零上一步的梯度，進行反向傳播和參數更新
    opt.zero_grad()
    loss.backward()
    opt.step()

    # 每10個epoch打印一次訓練準確率和損失
    if epoch % 10 == 0:
        pred_labels = torch.argmax(pred, dim=1)
        acc_ = 100.0 * (pred_labels == yt).sum().item() / batch_size
        print(f'Epoch {epoch}:')
        print(f'  Training Accuracy: {acc_:.2f}%')
        print(f'  Training Loss: {loss.item():.4f}')

# 定義測試函式
def test_acc(M):
    M.eval()  # 設置模型為評估模式
    with torch.no_grad():  # 禁用梯度計算
        xt = test_data.data.unsqueeze(1).float() / 255.0
        yt = test_data.targets
        preds = M(xt)
        pred_ind = torch.argmax(preds, dim=1)
        acc = (pred_ind == yt).sum().float() / len(test_data)
    return acc

# 保存模型
torch.save(M.state_dict(), "mnist.pt")

# 測試模型
M.load_state_dict(torch.load("mnist.pt", weights_only=True))
acc2 = test_acc(M)
print(f'Testing Accuracy: {(acc2 * 100).item():.2f}%')
