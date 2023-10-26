from datasets import load_from_disk
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

def tokenize(batch, tokenize_columun, tokenizer):
    """
    """
    tok = tokenizer(batch[tokenize_columun], truncation=True, padding=False)
    tok[f'{tokenize_columun}_ids'] = tok["input_ids"]
    return tok


def load_and_tokenize_dataset(
    data_dir,
    tokenize_columns:list,
    tokenizer
):

    dataset = load_from_disk(data_dir)
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={'tokenize_columun':tokenize_columns[0],
                    'tokenizer':tokenizer},
        batched=True,
        remove_columns=[tokenize_columns[0]]
    )
    tokenized_dataset = tokenized_dataset.map(
        tokenize,
        fn_kwargs={'tokenize_columun':tokenize_columns[1],
                    'tokenizer':tokenizer},
        batched=True,
        remove_columns=[tokenize_columns[1]]
    )

    return tokenized_dataset


def generate_attention_mask(tensor, tokenizer):
    attention_mask = (tensor != tokenizer.pad_token_id).float()
    return attention_mask


def collate_batch(batch, collate_columns, tokenizer, sts:bool=False):
    first_input      = [torch.LongTensor(record[collate_columns[0]])\
                                              for record in batch]
    second_input = [torch.LongTensor(record[collate_columns[1]])\
                                                for record in batch]
    first_input_ids   = pad_sequence(first_input, batch_first=True, padding_value=tokenizer.pad_token_id)
    second_input_ids   = pad_sequence(second_input, batch_first=True, padding_value=tokenizer.pad_token_id)
    #att masks
    first_input_att_mask = generate_attention_mask(first_input_ids, tokenizer)
    second_input_att_mask = generate_attention_mask(second_input_ids, tokenizer)
    
    if sts:
      return ([example[collate_columns[0]] for example in batch],
           [example[collate_columns[1]] for example in batch],
           first_input_ids, second_input_ids, torch.tensor([example['similarity_score'] \
                                                        for example in batch]),
          first_input_att_mask,
          second_input_att_mask)

    return first_input_ids, second_input_ids, \
            torch.LongTensor([record['label'] for record in batch]),\
            first_input_att_mask, \
            second_input_att_mask
    

class SBETOnli(nn.Module):
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


class SBETOsts(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, sentenceA, sentenceB, att_A, att_B):
        last_hidden_state_A = self.base_model(sentenceA)[0]
        last_hidden_state_B = self.base_model(sentenceB)[0]

        pooled_output_A = torch.mean(torch.matmul(att_A, last_hidden_state_A), dim=1)
        pooled_output_B = torch.mean(torch.matmul(att_B, last_hidden_state_B), dim=1)
        out = self.cos(pooled_output_A, pooled_output_B)
        return out