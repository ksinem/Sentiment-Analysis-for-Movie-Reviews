import torch
import torch.nn as nn
from torch.optim import Adam
import gensim.downloader as api


class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.wv_model = api.load("glove-wiki-gigaword-300")
        self.embedding = nn.Embedding(len(self.wv_model) + 1, 300)
        self.hidden_layer1 = nn.Linear(300, 256)
        self.hidden_layer2 = nn.Linear(256, 2)
        self.activation_func = nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.epochs = 5

    def forward(self, text):
        embedded_text = self.embedding(text).mean(dim=1)
        output = self.hidden_layer1(embedded_text)
        output = self.activation_func(output)
        output = self.hidden_layer2(output)
        return output

    def do_training_and_validation(self, train_dataloader, val_dataloader, train_dataset, val_dataset):
        self.train()
        for e in range(self.epochs):
            total_train_loss, total_correct_train = 0, 0
            for inputs, labels in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct_train += (predicted == labels).sum().item()
            train_accuracy = total_correct_train / len(train_dataset)
            print(
                f'Epoch {e + 1}/{self.epochs}: Training accuracy: {train_accuracy:.2%}'
            )
            self.eval()
            total_correct_val = 0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct_val += (predicted == labels).sum().item()
            val_accuracy = total_correct_val / len(val_dataset)
            print(
                f'Epoch {e + 1}/{self.epochs}: Validation accuracy: {val_accuracy:.2%} \n'
            )

    def do_testing(self, dataloader, dataset):
        self.eval()
        total_correct_test = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct_test += (predicted == labels).sum().item()
        test_accuracy = total_correct_test / len(dataset)
        print(
            f'Test accuracy: {test_accuracy:.2%}'
        )
