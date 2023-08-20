from utils import *
from models import *
import torch
import os
import pathlib
import pandas as pd

os.environ["WANDB_DISABLED"] = "true"
args = parse_train_args()
args.device = 'cuda'if torch.cuda.is_available() else 'cpu'
args.train_eval = 'train'
if __name__ == '__main__':
    dataset_dir = os.path.join(os.getcwd(),'datasets',args.dataset)
    train_data = pd.read_csv(dataset_dir+f'/process_train_{args.dataset_size}.csv', index_col=False)
    test_data = pd.read_csv(dataset_dir+f'/process_test_{args.dataset_size}.csv', index_col=False)
    model, tokenizer = get_model(args)
    
    if args.model in ['roberta-base','bert-base','distilroberta-base']:
        train_data, test_data = preprocess_huggingface(args, tokenizer, train_data, test_data)
        model = model.to(args.device)
        # try:
        #   model.from_pretrained(pathlib.PurePath(args.load_dir))
        #   print('Successfully load trained model')
        # except:
        #   print('Fail to load trained model')
        train_huggingface(args, model, train_data, test_data)
        model.save_pretrained(args.load_dir)
    else:
        train_data, test_data = preprocess_data(args, tokenizer, train_data, test_data)
        train_loader = construct_loader(args, train_data)
        test_loader = construct_loader(args, test_data)
        try:
            model.load_state_dict(torch.load(f'{args.load_dir}/best_model.pt', map_location=args.device))
            print('Successfully load trained model')
        except:
            print('Fail to load trained model')
        train(args,train_loader,test_loader,model)
        evaluate(args,train_loader,model)
