import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

class VGG(nn.Module):
    def __init__(self, in_channels, num_classes=10, dropout=False, batchnorm=False, p_dropout=0.2):

        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.p_dropout = p_dropout
        self.dropout = dropout
        # convolutional layers
        if dropout and not batchnorm:
            print("VGG model with dropout")
            self.conv_layers = nn.Sequential(

            # First VGG block
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(),  
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),  
            nn.Dropout(0.2),

            # Second VGG block
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),

            # Third VGG block
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                     padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                     padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),

        )
                
        # fully connected linear layers
            self.linear_layers = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=self.num_classes),
            nn.Softmax()
        )        
            
        if batchnorm and not dropout:
            print("VGG model with batchnorm")

            pass
        
        if batchnorm and dropout:
            print("VGG model with dropout and batchnorm")

            pass
        
        else:

            self.conv_layers = nn.Sequential(

            # First VGG block
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(),  
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),  

            # Second VGG block
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                     padding="same"),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),

            # Third VGG block
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                     padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                     padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
                
        # fully connected linear layers
            self.linear_layers = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.num_classes),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        x = self.linear_layers(x)
        return x

    def init_weights(self):
        torch.nn.init.he_uniform(self.conv_layers.weight)
        self.conv_layers.bias.data.fill_(0.01)
        torch.nn.init.he_uniform(self.linear_layers.weight)
        self.linear_layers.bias.data.fill_(0.01)

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, device, log_dir, obj_performance):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.obj_performance = obj_performance
        self.writer = SummaryWriter(log_dir=f"logs/" + log_dir + f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    def train(self, epochs):
        
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        
        for epoch in tqdm(range(epochs)):
            train_loss = 0.0
            train_acc = 0.0
            self.model.train()

            for i, data in enumerate(self.train_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item() 

            train_loss /= len(self.train_dataloader)
            train_acc /= len(self.train_dataloader.dataset)
            
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)

            val_loss = 0.0
            val_acc = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, data in enumerate(self.val_dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_acc += (predicted == labels).sum().item()

            val_loss /= len(self.val_dataloader)
            val_acc /= len(self.val_dataloader.dataset)
            
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)

            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
            
            if val_acc >= self.obj_performance:
                print("Solved")
        return train_loss_list, train_acc_list, val_loss_list, val_acc_list