import OpenAttack
import torch
import datasets
import tqdm
from contextlib import contextmanager
import numpy as np
from utils import *
from models import *
import os
import pathlib
import pandas as pd

os.environ["WANDB_DISABLED"] = "true"
@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request
    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})

args = parse_train_args(True)
# index = args.chunk
index = 70
args.train_eval_sample = 'train'
with open(f'generated_data/dataset_{index}.txt', 'r') as f:
    dataset = f.read()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_lst = {'amazon_review_full':5,
               'amazon_review_polarity':2,'dbpedia':14,
               'yahoo_answers':10,'ag_news':4,
               'yelp_review_full':5,'yelp_review_polarity':2,
               'banking77__2':2, 'banking77__4':4, 'banking77__5':5, 
               'banking77__10':10, 'banking77__14':14,
               'tweet_eval_emoji_2':2, 'tweet_eval_emoji_4':4, 'tweet_eval_emoji_5':5, 
               'tweet_eval_emoji_10':10, 'tweet_eval_emoji_14':14,
              }

with open(f'generated_data/dataset_{index}.txt','r') as f:
    args.dataset = f.read()

test_index = np.load(f'generated_data/test_index_{index}.npy')
train_index = np.load(f'generated_data/train_index_{index}.npy')
args.dataset, args.number_of_class = dataset, dataset_lst[dataset]
model, tokenizer = get_model(args)
train_data, test_data = load_dataset(args,train_index,test_index) # Dataframe

# Train (first time - for generating adversarial examples)
if args.model in ['roberta-base','bert-base','distilroberta-base']:
    train_data, test_data = preprocess_huggingface(args, tokenizer, train_data, test_data)
    model = model.to(args.device)
    if args.load_checkpoint == 'True':
        model.from_pretrained(pathlib.PurePath(args.load_dir))
    else:
        train_huggingface(args, model, train_data, train_data)
        model.save_pretrained(args.load_dir)

# Load test data
torch.cuda.empty_cache()
args.train_eval_sample = 'eval'
test_index = np.load(f'generated_data/{args.dataset}_test_index.npy')
attack_data = load_dataset(args,test_index=test_index)
attack_data = preprocess_huggingface(args, tokenizer, test_data=attack_data)
clsf = get_clsf(args, model, tokenizer)

# Generate Adversarial Examples
def generate_adversarial_examples(classifier, dataset, attacker = OpenAttack.attackers.PWWSAttacker()):
    attack_eval = OpenAttack.AttackEval(
        attacker,
        classifier,
    )
    correct_samples = [
        inst for inst in dataset if classifier.get_pred( [inst["x"]] )[0] == inst["y"]
    ]
    
    accuracy = len(correct_samples) / len(dataset)
    
    adversarial_samples = {
        "x_old": [],
        "x_new": [],
        "y": [],
    }
    
    for result in tqdm.tqdm(attack_eval.ieval(correct_samples), total=len(correct_samples)):
        if result["success"]:
            adversarial_samples["x_old"].append(result["data"]["x"])
            adversarial_samples["x_new"].append(result["result"])
            adversarial_samples["y"].append(result["data"]["y"])
    
    attack_success_rate = len(adversarial_samples["x_new"]) / len(correct_samples)
    print("Accuracy: %lf%%\nAttack success rate: %lf%%" % (accuracy * 100, attack_success_rate * 100))
    return adversarial_samples

adversarial_samples = {"x_new":[],"y":[]}
for attack_type in ['TextFooler','PWWS','DeepWordBug','BERT']:
    with no_ssl_verify():
        if attack_type == 'TextFooler':
            attacker = OpenAttack.attackers.TextFoolerAttacker()
        elif attack_type == 'PWWS':
            attacker = OpenAttack.attackers.PWWSAttacker()
        elif attack_type == 'DeepWordBug':
            attacker = OpenAttack.attackers.DeepWordBugAttacker()
        elif attack_type == 'BERT':
            attacker = OpenAttack.attackers.BERTAttacker()
        samples = generate_adversarial_examples(clsf,attack_data,attacker)
        adversarial_samples["x_new"] += samples["x_new"]
        adversarial_samples["y"] += samples["y"]

adversarial_samples['y'] = list(map(lambda x:x+1,adversarial_samples['y']))
pd.DataFrame({'label':adversarial_samples['y'], \
              'text':adversarial_samples['x_new']}) \
            .to_csv(args.custom_data_dir,index=False)