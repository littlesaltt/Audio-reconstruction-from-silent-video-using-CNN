import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL.Image as Image

# 利用 CNN 於 FashionMNIST 資料
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 建立 3 層捲積層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        # 池化層採 max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 建立 3 層全連接層
        self.fc1 = nn.Linear(in_features=128 * 11 * 11, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        # 設定 dropout 和激活函數 relu
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # first conv
        # input shape 1 * 28 * 28, output shape = 64 * 26 * 26, Note: (n-f+1), 28-3+1=26
        x = self.relu(self.conv1(x))
        # second conv
        # input shape 64 * 26 * 26, output shape = 64 * 24 * 24
        x = self.relu(self.conv2(x))
        # third conv
        # input shape 64 * 24 * 24, output shape = 128 * 22 * 22, after pooling -> 128 * 11 * 11, Note: (n-f)/s+1, (22-2)/2+1=11
        x = self.pool(self.relu(self.conv3(x)))

        # 全部做 flatten
        x = torch.flatten(x, 1)
        # fully connection layers
        # 可以決定要 dropout 的量、要對哪些層做 relu，都可以修改，都是實驗
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    # Device configuration
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # 設定超參數
    epochs = 50
    batch_size = 4
    lr = 0.01

    # 這次的資料集不在 local，所以要 import 資料集並設定成 Pytorch dataset
    # Dataset: FashionMNIST dataset
    # 可以選擇對資料做資料擴充
    # 這裡採取了 transforms.RandomRotation，也就是隨機角度旋轉圖片，將其轉化為 tensor 型態，然後做 normalize
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose(
                            [transforms.RandomRotation(degrees=20),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose(
                            [transforms.RandomRotation(degrees=20),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    # 取 80% 的資料做 validation 資料集
    train_data_size = int(0.8*len(train_data))
    val_data_size = len(train_data) - train_data_size

    # 透過 random_split 來隨機切 train_data 的資料到 validation data
    train_data, val_data = random_split(train_data, [train_data_size, val_data_size])

    # Dataloader 來 load 資料，在這裡設計 load 的 batch size，shuffle=True 代表要讀取隨機打散的資料
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # model 為我們建立的 CNN，loss function 是 CrossEntropyLoss，採取 SGD 隨機梯度下降法
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_validation_loss = float('inf')

    # 我們這次將 train loss 和 validation loss 都記錄下來，畫成折線圖
    all_training_loss = []
    all_validation_loss = []
    for epoch in range(epochs):
        # ----------------------------- training ----------------------------------------------- #
        # 訓練模型要設定成 model.train()
        model.train()
        # 從 train loader 讀 batch 跑多筆資料
        for i, (x, y) in enumerate(train_loader):
            # 利用 GPU 來做操作
            x = x.to(device)
            y = y.to(device)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward -> backward -> update
            # 清除所有梯度
            optimizer.zero_grad()
            # 將 train_data 依 batch size 的數量，放進模型訓練，將該筆資料的預測值存進 predict
            predict = model(x)
            # 計算 loss，把預測的 predict (也就是 ŷ)，和 y 去做 cross entropy 計算
            train_loss = criterion(predict, y)

            train_loss.backward() # 透過反向傳播獲得每個參數的梯度值
            optimizer.step()   # 透過梯度下降執行參數更新

        all_training_loss.append(train_loss.data.cpu())

        # ----------------------------- validation ----------------------------------------------- #
        # 評估模型要設定成 model.eval()
        model.eval()
        # torch.no_grad() 顧名思義就是 no gradient
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                if torch.cuda.is_available():
                  x = x.cuda()
                  y = y.cuda()

                predict = model(x)
                validation_loss = criterion(predict, y)

            all_validation_loss.append(validation_loss.data.cpu())

            # 儲存最好的權重
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model, "best_model.pth")
                print('New best model saved at epoch', epoch+1)


        print('Train Epoch: {}/{} Traing_Loss: {} Validation_Loss: {}'.format(epoch+1, epochs, train_loss.data, validation_loss.data))
    print('Finished Training')

    # ---------------------- comput accuracy -----------------------------------------------#
    correct_train, total_train = 0, 0
    # 計算 train accuracy 和 validation accuracy
    for i, (x, y) in enumerate(train_loader):
        # 利用 GPU 來做操作
        x = x.to(device)
        y = y.to(device)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # 將 train_data 依 batch size 的數量進行預測
        predict = model(x)
        predicted_label = torch.max(predict.data, 1)[1] # 預測 label
        total_train += len(y)    # 計算總共有多少筆 training data

        # 用 batch 的多筆資料，一次加總 predicted_label == y 成功的數量                                       #
        correct_train += (predicted_label == y).float().sum()

    # 計算訓練準確率
    train_accuracy = 100 * correct_train / float(total_train)

    correct_vaild, total_vaild = 0, 0
    for i, (x, y) in enumerate(val_loader):
        # 利用 GPU 來做操作
        x = x.to(device)
        y = y.to(device)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # 將 validation_data 依 batch size 的數量進行預測
        predict = model(x)
        predicted_label = torch.max(predict.data, 1)[1] # 預測 label
        total_vaild += len(y)    # 計算總共有多少筆 validation data

        # 用 batch 的多筆資料，一次加總 predicted_label == y 成功的數量
        correct_vaild += (predicted_label == y).float().sum()

    # 計算驗證準確率
    validation_accuracy = 100 * correct_vaild / float(total_vaild)

    # ----------------------------- draw loss picture -----------------------------
    # 畫出 Loss 的結果圖
    plt.plot(range(len(all_training_loss)), all_training_loss, 'indianred', label='Training_loss')
    plt.plot(range(len(all_validation_loss)), all_validation_loss, '#7eb54e', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    # plt.show()

    # ---------------------- Testing -----------------------------------------------#
    # testing 也是一種評估模型模式，一樣不用計算梯度
    model = torch.load("best_model.pth")
    model.eval()

    # 這次是從 FashionMNIST dataset 拿資料，有獲得了 test 的 label
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):

          x = x.to(device)
          y = y.to(device)

          predict = model(x)
          predicted_label = torch.max(predict.data, 1)[1] # 預測 label
          total_test += len(y)    # 計算總共有多少筆 test data

          # 用 batch 的多筆資料，一次加總 predicted_label == y 成功的數量
          correct_test += (predicted_label == y).float().sum()

    test_accuracy = 100 * correct_test / total_test


    # 輸出最終結果
    print()
    print("-------- Final Result -------")
    print("Epoch:", epochs, ", Learning Rate:", lr)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print("-------------------------------")

if __name__ == '__main__':
    train()