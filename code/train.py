from datasets import load_from_disk
import logging
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from torch import optim
from transformers import get_scheduler, AdamW
import os
import argparse
import sys
import smdebug
from smdebug.pytorch import Hook, SaveConfig
from smdebug import modes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



def tokenize(batch, tokenize_columun):
    """
    """
    tok = tokenizer(batch[tokenize_columun], truncation=True, padding=False)
    tok[f'{tokenize_columun}_ids'] = tok["input_ids"]
    return tok


def load_and_tokenize_dataset(
    data_dir
):

    dataset = load_from_disk(data_dir)
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={'tokenize_columun':'premise'},
        batched=True,
        remove_columns=['premise']
    )
    tokenized_dataset = tokenized_dataset.map(
        tokenize,
        fn_kwargs={'tokenize_columun':'hypothesis'},
        batched=True,
        remove_columns=['hypothesis']
    )

    return tokenized_dataset


def generate_attention_mask(tensor):
    global tokenizer
    attention_mask = (tensor != tokenizer.pad_token_id).float()
    return attention_mask


def collate_batch(batch):
    premise      = [torch.LongTensor(record['premise_ids'])\
                                              for record in batch]
    hypotesis = [torch.LongTensor(record['hypothesis_ids'])\
                                                for record in batch]
    prem_ids   = pad_sequence(premise, batch_first=True, padding_value=tokenizer.pad_token_id)
    hypo_ids   = pad_sequence(hypotesis, batch_first=True, padding_value=tokenizer.pad_token_id)
    #att masks
    prems_att_mask = generate_attention_mask(prem_ids)
    hypo_att_mask = generate_attention_mask(hypo_ids)

    return prem_ids, hypo_ids, torch.LongTensor([record['label'] for record in batch]), prems_att_mask, hypo_att_mask


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



def validation_step(model, dataloader, device, step, val_dataloader, optimizer)
    dev_results = validate(model, dataloader, val_dataloader, optimizer, device)
    dev=dev_results


def _inner_train(model, train_dataloader, val_dataloader, num_epochs=1, step=0, lr=2e-5):
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

            step += 1

        validation_step(model, val_dataloader, device, step, val_dataloader, optimizer)


def train(args):
    logger.info("Loading tokenizer...\n")
    global tokenizer
    global model_name
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loading pretrained model\n")
    model = AutoModel.from_pretrained(model_name)
    logger.info("Pretrained model loaded\n")

    logger.info("Fetching and tokenizing data for training")
    train_dataset = load_and_tokenize_dataset(
        args.train_data_dir
    )
    logger.info("Tokenizing data for training loaded\n")
    logger.info("Fetching and tokenizing data for eval")
    eval_dataset = load_and_tokenize_dataset(
        args.val_data_dir
    )
    logger.info("Tokenizing data for eval loaded\n")
    # logger.info("Fetching and tokenizing data for test")
    # test_dataset = load_and_tokenize_dataset(
    #     args.test_data_dir
    # )
    logger.info("Tokenizing data for test loaded\n")

    logger.info("Collating and padding")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_batch, shuffle=True)
    val_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=collate_batch, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_batch, shuffle=True)
    logger.info("Collating and padding finished\n")


    logger.info("Starting training")
    _inner_train(model, train_dataloader, val_dataloader, num_epochs=args.epochs, step=args.step, lr=args.lr)
    logger.info("Model trained successfully")


    logger.info("Removing unused checkpoints to save space in container")
    os.system(f"rm -rf {args.model_dir}/checkpoint-*/")
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    print('\n'*10)
    print('RUNNING')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val-data-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    # parser.add_argument("--test-data-dir", type=str,
    #                    default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--log-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--logging-strategy", type=str, default="epoch")
    train(parser.parse_args())