import os
from datasets import load_dataset
import random
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='banking77', help='Dataset name on Huggingface')
parser.add_argument('--dataset_config', type=str, default='', help='Name of dataset configuration')
parser.add_argument('--max_labels', type=int, default=77, help='The maximum number of labels of dataset configuration')
parser.add_argument('--n_labels', type=int, default=14, help='The number of labels that need to be redistributed')
args = parser.parse_args()
assert args.n_labels <= args.max_labels, 'The number of labels cannot exceed the maximum number of labels'
assert args.n_labels > 1, 'The number of redistributed labels must be greater than 1'

n_labels = args.n_labels
interval = args.max_labels//n_labels
labels = list(range(interval*n_labels))
random.shuffle(labels)
o2n_lb = {}
for new_lb in range(n_labels):
    for i in range(interval):
        o2n_lb[labels[new_lb*interval+i]] = new_lb+1

if args.dataset_config:
    dataset = load_dataset(args.dataset_name, args.dataset_config)
else:
    dataset = load_dataset(args.dataset_name)

train_data = pd.DataFrame(dataset['train'])
train_data = train_data[train_data['label'] < n_labels*interval]
train_data['label'] = train_data['label'].map(lambda x:o2n_lb[x])
train_data = train_data.reset_index(drop=True)

if 'validation' in dataset.column_names:
    test_data = pd.DataFrame({
                            'text': dataset['test']['text']+dataset['validation']['text'],
                            'label': dataset['test']['label']+dataset['validation']['label']
                            })
else:
    test_data = pd.DataFrame(dataset['test'])
test_data = test_data[test_data['label'] < n_labels*interval]
test_data['label'] = test_data['label'].map(lambda x:o2n_lb[x])
test_data = test_data.reset_index(drop=True)

dataset_dir = os.path.join(os.getcwd(),'datasets',f'{args.dataset_name}_{args.dataset_config}_{n_labels}')
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
train_data.to_csv(dataset_dir+f'/train.csv', header=False, index=False)
test_data.to_csv(dataset_dir+f'/test.csv', header=False, index=False)