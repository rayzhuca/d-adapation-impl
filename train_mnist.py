import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torchvision import datasets, transforms

from d_adam import DAdam

device = "mps"

class MnistModel(nn.Module):
    max_epoch = 10
    learning_rate = 1e-3

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 1 * 1, 64)
        self.act_1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

        self.losses = []
        self.to(device)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))  # 28 -> 14
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))  # 14 -> 7
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))  # 7 -> 3
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))  # 3 -> 1
        x = x.view(x.size(0), -1)  # flatten
        x = self.act_1(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = DAdam(self.parameters())
        loss_function = nn.CrossEntropyLoss()
        dataloader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
        self.batches_per_epoch = len(dataloader)

        for epoch in range(self.max_epoch):
            self.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                all_preds.extend(y_pred.argmax(1).detach().cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            avg_loss = total_loss / len(dataloader)
            self.losses.append(avg_loss)

            print(f'Epoch {epoch + 1}: Accuracy = {acc:.4f}, Loss = {avg_loss:.4f}')

    def test(self, X):
        self.eval()
        dataloader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
        y_pred = []

        with torch.no_grad():
            for (X_batch,) in dataloader:
                X_batch = X_batch.to(device)
                y_pred_batch = self.forward(X_batch)
                y_pred.append(y_pred_batch.cpu())

        y_pred = torch.cat(y_pred, dim=0)
        return y_pred.argmax(1)

    def run(self):
        print('--- Start Training ---')
        self.train_model(self.data['train']['X'], self.data['train']['y'])

        print('\n--- Start Testing ---')
        train_pred_y = self.test(self.data['train']['X'])
        test_pred_y = self.test(self.data['test']['X'])

        print("\nTraining Set Metrics:")
        print_metrics(self.data['train']['y'], train_pred_y)
        print("\nTest Set Metrics:")
        print_metrics(self.data['test']['y'], test_pred_y)

        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print(f"Macro Accuracy:  {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    model = MnistModel()
    model.data = {
        'train': {
            'X': train_dataset.data.unsqueeze(1).float() / 255.0,
            'y': train_dataset.targets
        },
        'test': {
            'X': test_dataset.data.unsqueeze(1).float() / 255.0,
            'y': test_dataset.targets
        }
    }

    model.run()
