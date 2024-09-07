import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def preprocess_function(
    texts,
    labels,
    tokenizer,
    model_config,
    max_length=512,
):
    """
    Preprocess the input text and labels for training.
    """

    texts = [str(text) for text in texts]
    # Tokenize the input text
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Initialize the binary matrix for labels (multilabel)
    num_labels = len(model_config.label2id)
    binary_matrix = np.zeros((len(labels), num_labels), dtype=int)

    # Convert label strings to binary vectors
    for i, label_list in enumerate(labels):
        for label in label_list:
            if label in model_config.label2id:
                binary_matrix[i, model_config.label2id[label]] = 1

    # Convert binary matrix to a torch tensor
    binary_labels = torch.tensor(binary_matrix, dtype=torch.float32)

    return encoding, binary_labels


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model_config):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.model_config = model_config

        # Preprocess the data
        self.encodings, self.binary_labels = preprocess_function(
            texts, labels, tokenizer, model_config
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return dictionary of tokenized inputs and binary labels
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.binary_labels[idx]
        return item


def get_train_val_loaders(
    train_df,
    val_df,
    tokenizer,
    model_config,
    batch_size,
):
    """
    Get dataloaders for training and validation sets.
    """
    train_dataloader = DataLoader(
        CustomDataset(
            train_df["text"].values,
            train_df["labels"].values,
            tokenizer,
            model_config,
        ),
        shuffle=True,
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(
        CustomDataset(
            val_df["text"].values,
            val_df["labels"].values,
            tokenizer,
            model_config,
        ),
        shuffle=False,
        batch_size=batch_size,
    )

    return train_dataloader, val_dataloader
