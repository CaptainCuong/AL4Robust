from utils import *
from models import *
import torch
import OpenAttack
from contextlib import contextmanager
import pathlib
import os
import pandas as pd
@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request

    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})

args = parse_train_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.train_eval = 'eval'
if __name__ == '__main__':
    dataset_dir = os.path.join(os.getcwd(),'datasets',args.dataset)
    test_data = pd.read_csv(dataset_dir+f'/process_test_{args.dataset_size}.csv', index_col=False)
    # test_data = load_dataset(args)
    model, tokenizer = get_model(args)
    
    if args.model in ['roberta-base','bert-base']:
        dataset = preprocess_huggingface(args, tokenizer, test_data=test_data)
        try:
            model.from_pretrained(pathlib.PurePath(args.load_dir))
            print('Successfully load trained model')
        except:
            raise Exception('Fail to load trained model')
        model = model.to(args.device)
    else:
        dataset = preprocess_data(args, tokenizer, test_data=test_data)
        try:
            model.load_state_dict(torch.load(f'{args.load_dir}/best_model.pt', map_location=args.device))
            print('Successfully load trained model')
        except:
            raise Exception('Fail to load trained model')
        model = model.to(args.device)
    clsf = get_clsf(args, model, tokenizer)
    with no_ssl_verify():
        attacker = OpenAttack.attackers.PWWSAttacker()
        attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
            OpenAttack.metric.Fluency(),
            OpenAttack.metric.GrammaticalErrors(),
            OpenAttack.metric.SemanticSimilarity(),
            OpenAttack.metric.EditDistance(),
            OpenAttack.metric.ModificationRate()
        ] )
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

