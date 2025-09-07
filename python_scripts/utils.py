import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
import re
import gensim.downloader as api
import json


def load_data():
    path = "../data/IMDB Dataset.csv"
    data = pd.DataFrame(pd.read_csv(path))
    X_train, X_test, y_train, y_test = train_test_split(data["review"], data["sentiment"],
                                                        test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_dataloader(x, y):
    dataset = TensorDataset(x, y)
    print("Generating dataloader objects...")
    return DataLoader(dataset, batch_size=50, shuffle=True)


def texts_to_indices(texts, word_to_idx, max_len=256):
    indices_list = []
    for text in tqdm.tqdm(texts):
        if not isinstance(text, str):
            print(f"Warnung: Nicht-Zeichenketten-Daten übersprungen: {type(text)}")
            continue
        processed_words = [re.sub(r'[^\w\s]', '', word).lower() for word in text.split()]
        indices = [word_to_idx[word] if word in word_to_idx else 0 for word in processed_words]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        indices_list.append(indices)
    return torch.tensor(indices_list, dtype=torch.long)







