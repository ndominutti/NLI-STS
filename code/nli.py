from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler, AdamW
from huggingface_hub import login
from tqdm.auto import tqdm
from typing import Dict
from torch import optim
import torch.nn as nn
import auxiliars
import argparse
import logging
import torch
import sys
import os


# Set up logger
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
    return auxiliars.collate_batch(batch, ["premise_ids", "hypothesis_ids"], tokenizer)


def get_predictions(batch: torch.Tensor, model: nn.Module, device: str) -> torch.tensor:
    """
    Use the model to get predictions on the batch data

    Args:
      batch(torch.Tensor): data batch to be processed
      model(nn.Module): base model
      device(str): 'cuda' or 'cpu'

    Returns:
      torch.tensor: prediction tensor
    """
    batch_prem, batch_hypo, batch_labels, batch_prem_att, batch_hypo_att = batch
    batch_prem = batch_prem.to(device)
    batch_hypo = batch_hypo.to(device)
    batch_prem_att = batch_prem_att.to(device)
    batch_hypo_att = batch_hypo_att.to(device)
    batch_labels = batch_labels.to(device)
    return model(batch_prem, batch_hypo, batch_prem_att, batch_hypo_att), batch_labels


def validate(
    model: nn.Module, val_dataloader: torch.utils.data.IterableDataset, device: str
) -> Dict[str, Dict[str, float]]:
    """
    Run validation process on val data

    Args:
      model(nn.Module): base model to process the data and get predictions
      val_dataloader(torch.utils.data.IterableDataset): dataloader containing the validation data
      device(str): 'cuda' or 'cpu'

    Returns:
      Dict[str, Dict[str, float]]: sklearn's classification report
    """
    sum_loss = 0.0
    num_batches = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for batch in val_dataloader:
            preds, batch_labels = get_predictions(batch, model, device)

            sum_loss += nn.functional.cross_entropy(preds, batch_labels.view(-1)).cpu()
            all_labels.append(batch_labels.detach())
            all_outputs.append(preds.detach())
            num_batches += 1
    # Get the class with the highest probability
    preds = torch.cat(all_outputs).argmax(1).cpu().numpy()
    gold = torch.cat(all_labels).cpu().numpy()

    ret = classification_report(gold, preds, output_dict=True)
    loss = sum_loss / num_batches
    ret["loss"] = loss.item()
    return ret


def validation_step(
    model: nn.Module,
    val_dataloader: torch.utils.data.IterableDataset,
    device: str,
    step: int,
    writer: SummaryWriter,
) -> None:
    """
    Run a single validation step. Useful to use inside the training process (at the end
    of every epoch)

    Args:
      model(nn.Module): base model to process the data and get predictions
      val_dataloader(torch.utils.data.IterableDataset): dataloader containing the validation data
      device(str): 'cuda' or 'cpu'
      step(int):
      writer(SummaryWriter): tensorboard SummaryWriter

    Return:
      None
    """
    dev_results = validate(model, val_dataloader, device)
    dev = dev_results
    for key in ["0", "1", "2"]:
        writer.add_scalar(
            "dev/" + key + " Macro F1", dev_results[key]["f1-score"], global_step=step
        )
    writer.add_scalar(
        "dev/macro_f1", dev_results["macro avg"]["f1-score"], global_step=step
    )
    writer.add_scalar("dev/loss", dev_results["loss"], global_step=step)
    writer.add_scalar("dev/accuracy", dev_results["accuracy"], global_step=step)


def _inner_train(
    model: nn.Module,
    train_dataloader: torch.utils.data.IterableDataset,
    val_dataloader: torch.utils.data.IterableDataset,
    num_epochs: int,
    step: int,
    lr: float,
    writer: SummaryWriter,
) -> nn.Module:
    """
    Perform the main training process. Uses cuda device if it's available.
    Applies an optimizer linear scheduler. Uses cross_entropy as loss function.

    Args:
      model(nn.Module): base model to process the data and get predictions
      train_dataloader(torch.utils.data.IterableDataset): dataloader containing the training data
      val_dataloader(torch.utils.data.IterableDataset): dataloader containing the validation data
      num_epochs(int): number of epochs to be ran
      step(int): step for the tensorboard writer
      lr(int): learning rate to be used
      writer(SummaryWriter): tensorboard SummaryWriter

    Return:
      model(nn.Module): trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = auxiliars.SBETOnli(model)
    model = model.to(device)
    num_training_steps = len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        model.train()
        print("*" * 10 + str(epoch) + "*" * 10)
        for batch in tqdm(train_dataloader):
            preds = get_predictions(batch, model)
            optimizer.zero_grad()
            loss, batch_labels = nn.functional.cross_entropy(preds, batch_labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/loss", loss, global_step=step)
            writer.add_scalar("train/lr", current_lr, global_step=step)
            step += 1
        validation_step(
            model, val_dataloader, device, step, val_dataloader, optimizer, writer
        )
    return model


def save_model(model: nn.Module, hf_save_path: str) -> None:
    """
    Push models into hugging face hub

    Args:
      model(nn.Module): model to be pushed
      hf_save_path(str): hugging face hub path

    Returns
      None
    """
    model.base_model.push_to_hub(hf_save_path)


def train(args):
    """
    Training wraper. In charge of:
    * Login into HF
    * Download tokenizer
    * Download model
    * Tokenize datasets
    * Prepare dataloaders
    * Run training
    * Write summaries
    * Push the trained model to HF
    """
    logger.info("Login into HF...\n")
    login(args.hf_token)
    logger.info("Loading tokenizer...\n")
    global tokenizer
    global model_name
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = args.max_len
    logger.info("Loading pretrained model\n")
    model = AutoModel.from_pretrained(model_name)
    logger.info("Pretrained model loaded\n")

    logger.info("Fetching and tokenizing data for training")
    tokenize_columns = ["premise", "hypothesis"]
    train_dataset = auxiliars.load_and_tokenize_dataset(
        args.train_data_dir, tokenize_columns, tokenizer
    )
    logger.info("Tokenizing data for training loaded\n")
    logger.info("Fetching and tokenizing data for eval")
    eval_dataset = auxiliars.load_and_tokenize_dataset(
        args.val_data_dir, tokenize_columns, tokenizer
    )
    logger.info("Collating and padding")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collator,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, collate_fn=collator, shuffle=True
    )
    logger.info("Collating and padding finished\n")
    logger.info("Starting training")
    writer = SummaryWriter("logs")
    model = _inner_train(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=args.epochs,
        step=args.step,
        lr=args.lr,
        writer=writer,
    )
    logger.info("Model trained successfully")
    logger.info("Pushing into HF")
    save_model(model, args.hf_save_path)
    logger.info("Pushed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--train-data-dir", type=str)
    parser.add_argument("--val-data-dir", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-save-path", type=str)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    train(parser.parse_args())
