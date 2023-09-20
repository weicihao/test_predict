import torch
import torch.nn as nn
import torch.optim as optim
import getNewData
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

def get_grad(modul, grad_input, grad_output):
    grads = []
    for grad in grad_input:
        if grad is not None:
            grads.append(grad.detach().mean().item())
    print(grads)

#计算准确率
def calcu_accur(train_data_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in train_data_loader:
            inputs = batch[0].to(device).float()
            outputs = model(inputs)
            _, predict = torch.max(outputs, dim = 1)
            total = total + inputs.size(0)
            correct = correct + (predict == inputs).sum().item()
    accuracy = correct / total
    return accuracy

# 获取数据
def getData(x):
    # 从Excel文件加载数据

    if x == 1:
        data_frame = pd.read_excel("D:\\桌面\\LSTM(1)\\NewDataC1(1).xlsx")
    elif x == 2:
        data_frame = pd.read_excel("C:\\Users\monai\Desktop\\NewDataC2.xlsx")
    elif x == 3:
        data_frame = pd.read_excel("C:\\Users\monai\Desktop\\NewDataC3.xlsx")
    else:
        return

    colData = data_frame['数据']
    list_data = list(colData)

    return list_data


data = getData(1)
min_val = min(data, key=float)
max_val = max(data)
nor_data = [(x - min_val) / (max_val - min_val) for x in data]

input_data = torch.tensor(nor_data, requires_grad=True).unsqueeze(1)

split_spot = 0.8  # 8 : 2 训练：测试
train_size = int(len(input_data) * split_spot)
train_data = input_data[:train_size]
train_data.requires_grad_(True)
test_data = input_data[train_size:]
test_data.requires_grad_(True)

print('train_data_length', len(train_data))
train_dataset = TensorDataset(train_data)
test_dataset = TensorDataset(test_data)

train_dataloder = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloder = DataLoader(test_dataset, batch_size=1, shuffle=True)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers ,bidirectional=True)

        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers ,bidirectional=True, dropout = 0.3)
        # self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.active = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.fc = nn.Sequential(
        #     x = nn.Linear(hidden_size * 2, hidden_size),
        #     x = nn.ReLU(x),
        #     x = nn.Linear(hidden_size, output_size)
        #     # nn.Tanh(),
        #     # nn.Linear(hidden_size, output_size)
        # )

    def forward(self, input):
        output, _ = self.lstm(input.view(len(input), 1, -1))
        # output = self.dropout(output)

        # output = self.fc(output.view(len(input), -1))

        output = self.fc1(output.view(len(input), -1))
        output = self.active(output)
        output = self.fc2(output)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
hidden_size = 128
output_size = 1
num_layers = 2

model = BiLSTM(input_size, hidden_size,num_layers, output_size).to(device)

criterion = nn.L1Loss().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=0.1)  
scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.7)
num_epochs = 50
for modul in model.modules:
    modul.register_backward_hook(get_grad)

model.train()
for epoch in range(num_epochs):
    sum_loss = 0
    for batch in train_dataloder:
        batch_data = batch[0].to(device).float()
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    avg_loss = sum_loss / len(train_dataloder)
    if (epoch + 1) % 10 == 0:
        scheduler.step()
        print('learn_rate:', optimizer.param_groups[0]['lr'])
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")

    # if (epoch + 1) % 13 == 0:
        # scheduler.step()
# 保存训练好了的模型，防止意外程序结束
# torch.save(model.state_dict(), "bi_model.pt")

# model = BiLSTM(input_size, hidden_size,num_layers, output_size).to(device)
# 加载训练模型
# model.load_state_dict(torch.load("bi_model.pt"))

model.eval()
with torch.no_grad():
    pre_val_train = model(train_data.to(device))
    pre_val_test = model(test_data.to(device))

print('准确率', calcu_accur(train_dataloder, model))

pre_val_train = pre_val_train.cpu().squeeze().tolist()
predicted_values_test = pre_val_test.cpu().squeeze().tolist()

pre_val_train = [x * (max_val - min_val) + min_val for x in pre_val_train]
pre_val_test = [x * (max_val - min_val) + min_val for x in pre_val_test]

print("Train_Pre:", len(pre_val_train))
print(pre_val_train[:100])


print("Test_Pre:", len(pre_val_test))
print(pre_val_test[:100])
