from datasets import load_from_disk
import torch
from torch.nn.utils.rnn import pad_sequence

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


def collate_batch(batch, collate_columns, tokenizer):
    first_input      = [torch.LongTensor(record[collate_columns[0]])\
                                              for record in batch]
    second_input = [torch.LongTensor(record[collate_columns[1]])\
                                                for record in batch]
    first_input_ids   = pad_sequence(first_input, batch_first=True, padding_value=tokenizer.pad_token_id)
    second_input_ids   = pad_sequence(second_input, batch_first=True, padding_value=tokenizer.pad_token_id)
    #att masks
    first_input_att_mask = generate_attention_mask(first_input_ids, tokenizer)
    second_input_att_mask = generate_attention_mask(second_input_ids, tokenizer)

    return first_input_ids, second_input_ids, torch.LongTensor([record['label'] for record in batch]), first_input_att_mask, second_input_att_mask