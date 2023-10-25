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
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def collator(batch):
  """
  """
  global tokenizer
  return auxiliars.collate_batch(batch, ['sentence1_ids', 'sentence2_ids'], tokenizer)





def sts_test(args):
    logger.info("Login into HF...\n")
    login(args.hf_token)
    logger.info("Loading tokenizer...\n")
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loading pretrained model\n")
    model = AutoModel.from_pretrained(model_name)
    logger.info("Pretrained model loaded\n")

    tokenize_columns = ['sentence1','sentence2']
    logger.info("Fetching and tokenizing data for test")
    eval_dataset = auxiliars.load_and_tokenize_dataset(
        args.test_data_dir,
        tokenize_columns,
        tokenizer
    )
    logger.info("Tokenizing data for test loaded\n")

    logger.info("Collating and padding")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collator, shuffle=True)
    logger.info("Collating and padding finished\n")

