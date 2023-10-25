import auxiliars
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from torch import optim
from transformers import get_scheduler, AdamW
import os
import argparse
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import login


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def collator(batch):
  """
  """
  global tokenizer
  return auxiliars.collate_batch(batch, ['premise_ids', 'hypothesis_ids'], tokenizer)

class SBETO(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fc = nn.Linear(768 * 3, 3)

    def forward(self, sentenceA, sentenceB, att_A, att_B):
        last_hidden_state_A = self.base_model(sentenceA)[0]
        last_hidden_state_B = self.base_model(sentenceB)[0]
        pooled_output_A = torch.mean(torch.matmul(att_A, last_hidden_state_A), dim=1)
        pooled_output_B = torch.mean(torch.matmul(att_B, last_hidden_state_B), dim=1)
        diff = torch.abs(pooled_output_A - pooled_output_B)
        concatenated = torch.cat([pooled_output_A, pooled_output_B, diff], dim=1)
        out = self.fc(concatenated)
        return out


def validate(model, dataloader, val_dataloader, optimizer, device):
    sum_loss = .0
    num_batches = 0

    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch_prem, batch_hypo, batch_labels, batch_prem_att, batch_hypo_att = batch
            batch_prem = batch_prem.to(device)
            batch_hypo = batch_hypo.to(device)
            batch_prem_att = batch_prem_att.to(device)
            batch_hypo_att = batch_hypo_att.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            preds = model(batch_prem, batch_hypo, batch_prem_att, batch_hypo_att)

            sum_loss += nn.functional.cross_entropy(
                preds,
                batch_labels.view(-1)
            ).cpu()
            all_labels.append(batch_labels.detach())
            all_outputs.append(preds.detach())

            num_batches += 1
    # calculo argmax para quedarme con la clase más probable
    preds = torch.cat(all_outputs).argmax(1).cpu().numpy()
    gold = torch.cat(all_labels).cpu().numpy()

    # Dame reporte de clasificación
    ret = classification_report(gold, preds, output_dict=True)

    #Convierto a los nombres posta
    loss = sum_loss / num_batches
    ret["loss"] = loss.item()
    return ret



def validation_step(model, dataloader, device, step, val_dataloader, optimizer, writer):
    dev_results = validate(model, dataloader, val_dataloader, optimizer, device)
    dev=dev_results
    for key in ["0", "1", "2"]:
        writer.add_scalar("dev/" + key+ " Macro F1", dev_results[key]["f1-score"], global_step=step)
    writer.add_scalar("dev/macro_f1", dev_results['macro avg']["f1-score"], global_step=step)
    writer.add_scalar("dev/loss", dev_results["loss"], global_step=step)
    writer.add_scalar("dev/accuracy", dev_results["accuracy"], global_step=step)


def _inner_train(model, train_dataloader, val_dataloader, num_epochs, step, lr, writer):
  """
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  model = SBETO(model)
  model = model.to(device)
  num_training_steps = len(train_dataloader)


  optimizer = AdamW(model.parameters(), lr=lr)
  scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps = num_training_steps // 10, num_training_steps=num_training_steps,)


  for epoch in range(num_epochs):
      model.train()
      print('*'*10 + str(epoch) + '*'*10)
      for batch in tqdm(train_dataloader):
          batch_prem, batch_hypo, batch_labels, batch_prem_att, batch_hypo_att = batch
          batch_prem = batch_prem.to(device)
          batch_hypo = batch_hypo.to(device)
          batch_prem_att = batch_prem_att.to(device)
          batch_hypo_att = batch_hypo_att.to(device)
          batch_labels = batch_labels.to(device)
          optimizer.zero_grad()
          preds = model(batch_prem, batch_hypo, batch_prem_att, batch_hypo_att)
          loss = nn.functional.cross_entropy(
                    preds,
                    batch_labels.view(-1)
                )

          loss.backward()
          optimizer.step()

          scheduler.step()
          current_lr = scheduler.get_last_lr()[0]

          writer.add_scalar('train/loss', loss, global_step=step)
          writer.add_scalar('train/lr', current_lr, global_step=step)

          step += 1

      validation_step(model, val_dataloader, device, step, val_dataloader, optimizer, writer)
  return model

def save_model(model, hf_save_path):
  model.base_model.push_to_hub(hf_save_path)


def train(args):
    logger.info("Login into HF...\n")
    login(args.hf_token)
    logger.info("Loading tokenizer...\n")
    global tokenizer
    global model_name
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loading pretrained model\n")
    model = AutoModel.from_pretrained(model_name)
    logger.info("Pretrained model loaded\n")

    logger.info("Fetching and tokenizing data for training")
    tokenize_columns = ['premise','hypothesis']
    train_dataset = auxiliars.load_and_tokenize_dataset(
        args.train_data_dir,
        tokenize_columns,
        tokenizer
    )
    logger.info("Tokenizing data for training loaded\n")
    logger.info("Fetching and tokenizing data for eval")
    eval_dataset = auxiliars.load_and_tokenize_dataset(
        args.val_data_dir,
        tokenize_columns,
        tokenizer
    )
    logger.info("Tokenizing data for eval loaded\n")
    # logger.info("Fetching and tokenizing data for test")
    # test_dataset = load_and_tokenize_dataset(
    #     args.test_data_dir
    # )
    logger.info("Tokenizing data for test loaded\n")

    logger.info("Collating and padding")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collator, shuffle=True)
    val_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=collator, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_columns, shuffle=True)
    logger.info("Collating and padding finished\n")


    logger.info("Starting training")
    writer = SummaryWriter("logs")
    model = _inner_train(model, train_dataloader, val_dataloader, num_epochs=args.epochs, step=args.step, lr=args.lr, writer=writer)
    logger.info("Model trained successfully")
    logger.info("Pushing into HF")
    save_model(model, args.hf_save_path)
    logger.info("Pushed successfully")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--train-data-dir", type=str)
    parser.add_argument("--val-data-dir", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-save-path", type=str)
    # parser.add_argument("--test-data-dir", type=str)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    train(parser.parse_args())