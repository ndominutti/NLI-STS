# NLI-STS
This repository is an implementation of the sentence BERT model as seen in [Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT:Sentence Embeddings using SiameseBERT-Networks. https://arxiv.org/abs/1908.10084], but using spanish as the main language, that's the reason why instead of using a BERT model, a BETO model is used.

The main idea behind sentence BERT is to finetune a transformer based model in an NLI (Natural Language Inference) task and then use it to get embeddings from two sentences from a STS (Semantic Textual Similarity) and get a similarity score (as for example cosine distance).
The final testing is done between the similarity predicted by the model and the ground truth real similarity. A Spearman's correlation score is used for this.

---

### Repo structure
In this repo you will find
- requirements.txt 
- src/
  - auxiliars.py: file containing auxiliar functions and classes
  - download_datasets.py: setup file to download the data used from HuggingFace (https://huggingface.co/datasets/xnli & https://huggingface.co/datasets/stsb_multi_mt)
  - nli.py      : file that performs the NLI training process and uploads the model to HuggingFace
  - sts.py      : file that downloads the model from HuggingFace, performs the STS process and compare the score vs a base BETO model

---

### Usage
The easier way to use this repository is running from console the next commands:

SETUP
```
git clone https://github.com/ndominutti/NLI-STS.git
cd NLI-STS
pip3 install -r requirements.txt
python3 code/download_datasets.py
```

NLI
```
python3 code/train.py --model-name dccuchile/bert-base-spanish-wwm-cased --train-data-dir data/nli/train/ --val-data-dir data/nli/eval/ --hf-token <YOUR-HUGGINGFACE-TOKEN> --hf-save-path <YOUR-HUGGINGFACE-PATH>
```

STS
```
python3 code/sts.py --data-dir data/sts/ --hf-token <YOUR-HUGGINGFACE-TOKEN> --hf-load-path <YOUR-HUGGINGFACE-PATH>
```

---

### Results
In this implementation, I compare a BETO model without finetune in an NLI task the one that was finetuned. To compare them we use the spearman's correlation coeff between each model's prediction and the ground truth. The achieved results were:

| Model      | Spearman's correlation coeff |
|-----------|-----|
| SPEARMAN CORRELATION VANILLA SBETO     | 0.228  | 
| SPEARMAN CORRELATION SBETO FINETUNNED      | 0.597  |

Showing that finetunning in an NLI task considerably increases the model's performance in an STS task


