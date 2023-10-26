from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel, get_scheduler, AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import optim
import numpy as np
from tqdm import tqdm
import auxiliars
import argparse
import logging
import sys
from huggingface_hub import login

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def collator(batch):
  """
  """
  global tokenizer
  return auxiliars.collate_batch(batch, ['sentence1_ids', 'sentence2_ids'], tokenizer, True)


def _get_ranks(x: torch.Tensor, device) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp, device=device)
    ranks[tmp] = torch.arange(len(x), device=device)
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor, device):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x, device)
    y_rank = _get_ranks(y, device)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


def print_results(spearman_corr:list):
  """
  """
  index = ['SPEARMAN CORRELATION VANILLA SBETO',
         'SPEARMAN CORRELATION SBETO FINETUNNED']
  values = [spearman_corr[0], spearman_corr[1]]
  df = pd.DataFrame(values, index=index, columns=['SPEARMAN CORRELATION IN STS TASK'])
  print(df)


def _comparisson(dataloader, model_base, model_finetuned):
  """
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  modelos = [auxiliars.SBETOsts(model_base), auxiliars.SBETOsts(model_finetuned)]
  spearman_corr = []
  for model in modelos:
    model = model.to(device)
    corr_batch = []

    for batch in tqdm(dataloader):
        _, _, batch_sent1, batch_sent2, batch_labels, batch_sent1_att, batch_sent2_att = batch
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

    tokenize_columns = ['sentence1','sentence2']
    logger.info("Fetching and tokenizing data for test")
    dataset = auxiliars.load_and_tokenize_dataset(
        args.data_dir,
        tokenize_columns,
        tokenizer
    )
    logger.info("Tokenizing data for test loaded\n")

    logger.info("Collating and padding")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    logger.info("Collating and padding finished\n")

    logger.info("Running comparisson")
    _comparisson(dataloader, model_base, model_finetuned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-model-name", type=str, 
                              default='dccuchile/bert-base-spanish-wwm-cased')
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-load-path", type=str)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    sts_test(parser.parse_args())
