import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


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
        padding=False,
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


def custom_collate_fn(batch, tokenizer):
    """
    Custom collation function for handling padded sequences.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # Pad sequences to the maximum length of the batch
    padded_input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    # Convert labels to tensor
    labels = torch.stack(labels)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": labels,
    }


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
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
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
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        shuffle=False,
        batch_size=batch_size,
    )

    return train_dataloader, val_dataloader
