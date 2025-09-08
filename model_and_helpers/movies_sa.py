from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import tqdm
from utils import (
    load_data,
    texts_to_indices)
from neural_net import NeuralNet
import gensim.downloader as api


def run_skmodel(model):
    print("Loading and splitting the data into train test...")
    X_train_, X_test_, y_train_, y_test_ = load_data()
    print("Transforming the data...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(tqdm.tqdm(X_train_))
    X_test_vec = vectorizer.transform(tqdm.tqdm(X_test_))
    print(f"Initializing the {model} model...")
    model.fit(X_train_vec, y_train_)
    print("Making predictions and calculating accuracy...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test_, y_pred)

    print(f"Accuracy of the model: {accuracy}")


def run_pytorch_model(nn_model):

    X_train, X_temp, y_train, y_temp = load_data()
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
    word_to_idx = {word: idx + 1 for idx, word in enumerate(nn_model.wv_model.index_to_key)}

    datasets_X = [X_train, X_val]
    datasets_y = [y_train, y_val]
    tensor_datasets = []

    for (X, y) in zip(datasets_X, datasets_y):
        X_idx = texts_to_indices(X, word_to_idx)
        y_tensor = torch.tensor(y.map({'positive': 1, 'negative': 0}).values, dtype=torch.long)
        tensor_datasets.append(TensorDataset(X_idx, y_tensor))

    train_dataset = tensor_datasets[0]
    val_dataset = tensor_datasets[1]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    nn_model.do_training_and_validation(train_dataloader=train_loader,
                                        train_dataset=train_dataset,
                                        val_dataloader=val_loader,
                                        val_dataset=val_dataset)


def run_pytorch_model_test(nn_model, X_test_, y_test_):
    word_to_idx = {word: idx + 1 for idx, word in enumerate(nn_model.wv_model.index_to_key)}

    X_idx = texts_to_indices(X_test_, word_to_idx)
    y_tensor = torch.tensor(y_test_.map({'positive': 1, 'negative': 0}).values, dtype=torch.long)
    test_dataset = TensorDataset(X_idx, y_tensor)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    nn_model.do_testing(dataloader=test_loader, dataset=test_dataset)


if __name__ == "__main__":
    # run_skmodel(model=SVC(random_state=42))
    run_pytorch_model(nn_model=NeuralNet())
    # run_pytorch_model_test(nn_model=NeuralNet())
