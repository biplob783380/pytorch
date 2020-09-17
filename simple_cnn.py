# Reference video : https://www.youtube.com/watch?v=wnK3uWv_WkU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=4

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# create fully connected Network
# class NN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
    
# CNN network    
class NN(nn.Module):
    def __init__(self, input_size=1, num_classes=10):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        return x
        
    
    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 100
num_epoch = 1


# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = NN(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network:
for epoch in range(num_epoch):
    for idx, (data, label) in enumerate(train_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        # data = data.reshape(batch_size, -1) # no need for cnn
        predict = model(data)
        loss = criterion(predict, label) # predict=confident score of each class, lable= the right class index;
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'{epoch} has complete')


def check_accuracy(loader, model): 
    num_correct = 0
    num_sample = 0
    model.eval() # lets go to evaluate mode;
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
#             x = x.reshape(batch_size, -1) # no need for cnn
            
            score = model(x)
            _, prediction = score.max(dim=1)
            num_correct += (prediction==y).sum()
            num_sample += prediction.size(dim=0)
            
        print(f'Got {num_correct}/{num_sample} with accuracy {float(num_correct)/float(num_sample)*100:.2f}')
    model.train() # set back to train mode
            
        
check_accuracy(train_loader, model)      
check_accuracy(test_loader, model) 
