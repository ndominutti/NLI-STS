from transformers import AutoTokenizer, AutoModel, get_scheduler, AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import login
from datasets import load_dataset
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import auxiliars
import argparse
import logging
import torch
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def collator(batch: torch.Tensor):
    """
    Wrapper function around auxiliars.collate_batch.
    collator allow us to map the collate functionality to the dataset using the
    DataLoader class from torch

    Args:
      batch(torch.Tensor): data batch sent by the DataLoader class
    """
    global tokenizer
    return auxiliars.collate_batch(
        batch, ["sentence1_ids", "sentence2_ids"], tokenizer, True
    )


def _get_ranks(x: torch.Tensor, device: str) -> torch.Tensor:
    """
    Calculate the ranks of elements in the input tensor.

    Args:
      x(torch.Tensor): Input tensor containing elements to be ranked.
      device(str): 'cuda' or 'cpu'

    Returns:
      torch.Tensor: A tensor of the same shape as `x` containing the ranks of elements.

    Note:
    The returned ranks are zero-based, where 0 represents the smallest element in `x`.
    """
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp, device=device)
    ranks[tmp] = torch.arange(len(x), device=device)
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    """
    Compute correlation between 2 1-D vectors

    Args:
      x(torch.Tensor): Shape (N, )
      y(torch.Tensor): Shape (N, )
      device(str): 'cuda' or 'cpu'

    Returns:
      float: correlation coefficient
    """
    x_rank = _get_ranks(x, device)
    y_rank = _get_ranks(y, device)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n**2 - 1.0)
    return 1.0 - (upper / down)


def print_results(spearman_corr: list) -> None:
    """
    Print Spearman correlation results in a tabular format.

    Args:
      spearman_corr(list): A list containing two Spearman correlation values.

    Returns:
      None
    """
    index = [
        "SPEARMAN CORRELATION VANILLA SBETO",
        "SPEARMAN CORRELATION SBETO FINETUNNED",
    ]
    values = [spearman_corr[0], spearman_corr[1]]
    df = pd.DataFrame(values, index=index, columns=["SPEARMAN CORRELATION IN STS TASK"])
    print(df)


def _comparisson(
    dataloader: torch.utils.data.IterableDataset,
    model_base: nn.Module,
    model_finetuned: nn.Module,
) -> None:
    """
    Perform the main comparison process. Uses cuda device if it's available.
    Saves spearman correlation between the predictions and the ground truth values.
    Prints final spearman correlation score.

    Args:
      dataloader(torch.utils.data.IterableDataset): dataloader containing the comparison STS data
      model_base(nn.Module): base model to process the data and get predictions
      model_finetuned(nn.Module): finetuned model to process the data and get predictions

    Returns:
      None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = [auxiliars.SBETOsts(model_base), auxiliars.SBETOsts(model_finetuned)]
    spearman_corr = []
    for model in modelos:
        model = model.to(device)
        corr_batch = []
        for batch in tqdm(dataloader):
            (
                _,
                _,
                batch_sent1,
                batch_sent2,
                batch_labels,
                batch_sent1_att,
                batch_sent2_att,
            ) = batch
            batch_sent1 = batch_sent1.to(device)
            batch_sent2 = batch_sent2.to(device)
            batch_sent1_att = batch_sent1_att.to(device)
            batch_sent2_att = batch_sent2_att.to(device)
            batch_labels = batch_labels.to(device)
            preds = model(batch_sent1, batch_sent2, batch_sent1_att, batch_sent2_att)
            sp_corr = spearman_correlation(preds, batch_labels, device)
            corr_batch.append(sp_corr.cpu())
        spearman_corr.append(np.mean(corr_batch))
    print_results(spearman_corr)


def sts_test(args):
    """
    STS comparison wrapper. In charge of:
    * Login into HF
    * Download tokenizer
    * Download models
    * Tokenize datasets
    * Prepare dataloaders
    * Run comparison and print Spearman's correlation matrix
    """
    logger.info("Login into HF...\n")
    login(args.hf_token)
    logger.info("Loading finetuned model\n")
    model_finetuned = AutoModel.from_pretrained(args.hf_load_path)
    logger.info("Finetuned model loaded\n")
    logger.info("Loading tokenizer...\n")
    model_name = args.bert_model_name
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    logger.info("Loading BERT model\n")
    model_base = AutoModel.from_pretrained(args.bert_model_name)
    logger.info("BERT model loaded\n")
    tokenize_columns = ["sentence1", "sentence2"]
    logger.info("Fetching and tokenizing data for test")
    dataset = auxiliars.load_and_tokenize_dataset(
        args.data_dir, tokenize_columns, tokenizer
    )
    logger.info("Tokenizing data for test loaded\n")

    logger.info("Collating and padding")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True
    )
    logger.info("Collating and padding finished\n")
    logger.info("Running comparison")
    _comparisson(dataloader, model_base, model_finetuned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert-model-name", type=str, default="dccuchile/bert-base-spanish-wwm-cased"
    )
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-load-path", type=str)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    sts_test(parser.parse_args())
