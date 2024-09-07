import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def create_model(
    model_ckpt: str = "bert-base-uncased",
    device: torch.device = torch.device("cpu"),
    max_length: int = 512,
    unique_labels: list = [],
    idx2label: dict = {},
    label2idx: dict = {},
):
    """
    Creates a BERT model with a custom classification head.
    Args:
        model_ckpt (str, optional): The checkpoint of the pre-trained BERT model. Defaults to 'bert-base-uncased'.
        device (torch.device, optional): The device to use for the model. Defaults to torch.device('cpu').
        max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
        unique_labels (list): The unique labels in the dataset.
        idx2label (dict): A dictionary mapping index to label.
        label2idx (dict): A dictionary mapping label to index.
    Returns:
        torch.nn.Module: The BERT model with a custom classification head.
        AutoTokenizer: The tokenizer used to preprocess the input data.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt,
        max_length=max_length,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=len(unique_labels),
        id2label=idx2label,
        label2id=label2idx,
    ).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay_params = ["bias", "LayerNorm.weight"]
    param_names = [n for n, p in param_optimizer]
    param_values = [p for n, p in param_optimizer]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(ndp in n for ndp in no_decay_params)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if any(ndp in n for ndp in no_decay_params)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    return model, tokenizer, optimizer_grouped_parameters
