from datasets import load_dataset
import argparse


def download_nli(data_path):
  train = load_dataset("xnli", "es", split='train')
  val   = load_dataset("xnli", "es", split='validation')

  train.save_to_disk(f'{data_path}/nli/train') 
  val.save_to_disk(f'{data_path}/nli/eval')

def download_sts(data_path):
    test = load_dataset("stsb_multi_mt", "es", split='test')
    test.save_to_disk(f'{data_path}/sts')


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-path", type=str, default='data')
  args = parser.parse_args()
  
  print('Downloading NLI data...')
  download_nli(args.data_path)
  print('Downloadede NLI data successfuly!')
  print('Downloading STS data...')
  download_sts(args.data_path)
  print('Downloadede STS data successfuly!')