from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from datasets import Dataset
import torch.nn as nn
import torch


def tokenize(batch, tokenize_columun: str, tokenizer):
    """
    Apply tokenizer to the batch. Won't use default padding as a custom
    one is used in collation.

    Args:
      batch():
      tokenize_columun(str): name of the column to be used
      tokenizer(): AutoTokenizer object

    Returns:
      tokenized batch
    """
    tok = tokenizer(batch[tokenize_columun], truncation=True, padding=False)
    tok[f"{tokenize_columun}_ids"] = tok["input_ids"]
    return tok


def load_and_tokenize_dataset(
    data_dir: str, tokenize_columns: list, tokenizer
) -> Dataset:
    """
    Mapping process to apply tokenizer to the whole dataset.

    Args:
      data_dir(str): path to the data directory
      tokenize_columuns(list): names of the columns to be tokenized
      tokenizer(): AutoTokenizer object

      Returns:
        tokenized dataset
    """
    dataset = load_from_disk(data_dir)
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenize_columun": tokenize_columns[0], "tokenizer": tokenizer},
        batched=True,
        remove_columns=[tokenize_columns[0]],
    )
    tokenized_dataset = tokenized_dataset.map(
        tokenize,
        fn_kwargs={"tokenize_columun": tokenize_columns[1], "tokenizer": tokenizer},
        batched=True,
        remove_columns=[tokenize_columns[1]],
    )
    return tokenized_dataset


def generate_attention_mask(tensor: torch.tensor, tokenizer) -> torch.tensor:
    """
    Generate the attention mask using the values where the tensor
    is not a padding id.

    Args:
      tensor(torch.tensor): tensor to be processed
      tokenizer:

    Returns:
      torch.tensor: attention mask
    """
    attention_mask = (tensor != tokenizer.pad_token_id).float()
    return attention_mask


def collate_batch(
    batch: torch.Tensor, collate_columns: list, tokenizer, sts: bool = False
):
    """
    Collate and preprocess a batch of data for neural network input.

    This function takes a batch of data and preprocess it for use with a neural network.
    It tokenizes and pads input sequences and generates attention masks. The function is
    designed to be used in combination with PyTorch's DataLoader.

    Args:
      batch (torch.Tensor): A batch of data, where each element is a dictionary containing the input
                    sequences and other relevant information.
      collate_columns (list): A list of column names from the batch's dictionary, specifying
                              which columns contain the input sequences.
      tokenizer: A Hugging Face Transformers tokenizer used for tokenization.
      sts (bool): Whether the task is Semantic Textual Similarity (STS). Set to False if
                  an NLI training is being performed.

    Returns:
      If sts is False:
      - first_input_ids (torch.Tensor): Tokenized and padded input IDs for the first sequence.
      - second_input_ids (torch.Tensor): Tokenized and padded input IDs for the second sequence.
      - labels (torch.Tensor): Target labels for the batch.
      - first_input_att_mask (torch.Tensor): Attention mask for the first input.
      - second_input_att_mask (torch.Tensor): Attention mask for the second input.

      If sts is True:
      - first_inputs (list of torch.Tensor): Unprocessed first input sequences.
      - second_inputs (list of torch.Tensor): Unprocessed second input sequences.
      - first_input_ids (torch.Tensor): Tokenized and padded input IDs for the first sequence.
      - second_input_ids (torch.Tensor): Tokenized and padded input IDs for the second sequence.
      - similarity_scores (torch.Tensor): Ground truth similarity scores.
      - first_input_att_mask (torch.Tensor): Attention mask for the first input.
      - second_input_att_mask (torch.Tensor): Attention mask for the second input.
    """
    first_input = [torch.LongTensor(record[collate_columns[0]]) for record in batch]
    second_input = [torch.LongTensor(record[collate_columns[1]]) for record in batch]
    first_input_ids = pad_sequence(
        first_input, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    second_input_ids = pad_sequence(
        second_input, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # att masks
    first_input_att_mask = generate_attention_mask(first_input_ids, tokenizer)
    second_input_att_mask = generate_attention_mask(second_input_ids, tokenizer)

    if sts:
        return (
            [example[collate_columns[0]] for example in batch],
            [example[collate_columns[1]] for example in batch],
            first_input_ids,
            second_input_ids,
            torch.tensor([example["similarity_score"] for example in batch]),
            first_input_att_mask,
            second_input_att_mask,
        )

    return (
        first_input_ids,
        second_input_ids,
        torch.LongTensor([record["label"] for record in batch]),
        first_input_att_mask,
        second_input_att_mask,
    )


class SBETOnli(nn.Module):
    """
    Sentence-BERT like training implementation as seen in
    Nils Reimers and Iryna Gurevych. 2019.
    Sentence-BERT:Sentence Embeddings using SiameseBERT-Networks.
    https://arxiv.org/abs/1908.10084
    """

    def __init__(self, base_model: nn.Module):
        """
        Args:
          base_model(nn.Module): base pretrained transformer model to be used
        """
        super().__init__()
        self.base_model = base_model
        self.fc = nn.Linear(768 * 3, 3)

    def forward(
        self,
        sentenceA: torch.Tensor,
        sentenceB: torch.Tensor,
        att_A: torch.Tensor,
        att_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the Sentence BERT model.

        This method takes input sentences, their attention masks, and computes sentence embeddings
        using the base model. It then calculates the difference and concatenation of the
        pooled sentence embeddings, passes them through a fully connected layer,
        and returns the output (as a siamese network).

        Args:
          sentenceA (torch.Tensor): Input embeddings for sentence A.
          sentenceB (torch.Tensor): Input embeddings for sentence B.
          att_A (torch.Tensor): Attention weights for sentence A.
          att_B (torch.Tensor): Attention weights for sentence B.

        Returns:
          torch.Tensor: Output tensor after processing through the Sentence BERT model.
        """
        last_hidden_state_A = self.base_model(sentenceA)[0]
        last_hidden_state_B = self.base_model(sentenceB)[0]
        pooled_output_A = torch.mean(torch.matmul(att_A, last_hidden_state_A), dim=1)
        pooled_output_B = torch.mean(torch.matmul(att_B, last_hidden_state_B), dim=1)
        diff = torch.abs(pooled_output_A - pooled_output_B)
        concatenated = torch.cat([pooled_output_A, pooled_output_B, diff], dim=1)
        out = self.fc(concatenated)
        return out


class SBETOsts(nn.Module):
    """
    Sentence-BERT like STS test implementation as seen in
    Nils Reimers and Iryna Gurevych. 2019.
    Sentence-BERT:Sentence Embeddings using SiameseBERT-Networks.
    https://arxiv.org/abs/1908.10084
    """

    def __init__(self, base_model: nn.Module):
        """
        Args:
          base_model(nn.Module): base pretrained transformer model to be used
        """
        super().__init__()
        self.base_model = base_model
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(
        self,
        sentenceA: torch.Tensor,
        sentenceB: torch.Tensor,
        att_A: torch.Tensor,
        att_B: torch.Tensor,
    ):
        """
        Perform the forward pass of the STS implementation of Sentence BERT.

        This method takes input sentences, their attention masks, and computes sentence embeddings
        with the pretrained model in NLI. It then calculates the cosine similarity between the pooled
        sentence embeddings and returns a similarity score, indicating the semantic similarity between
        the input sentences.

        Args:
          sentenceA (torch.Tensor): Input embeddings for sentence A.
          sentenceB (torch.Tensor): Input embeddings for sentence B.
          att_A (torch.Tensor): Attention weights for sentence A.
          att_B (torch.Tensor): Attention weights for sentence B.

        Returns:
          torch.Tensor: Similarity score indicating the semantic similarity between the input sentences.
        """
        last_hidden_state_A = self.base_model(sentenceA)[0]
        last_hidden_state_B = self.base_model(sentenceB)[0]

        pooled_output_A = torch.mean(torch.matmul(att_A, last_hidden_state_A), dim=1)
        pooled_output_B = torch.mean(torch.matmul(att_B, last_hidden_state_B), dim=1)
        out = self.cos(pooled_output_A, pooled_output_B)
        return out
