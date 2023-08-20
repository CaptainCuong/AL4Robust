from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_model(args):
    if args.model == 'roberta-base':
        return AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('roberta-base',model_max_length=args.max_length)
    elif args.model == 'distilroberta-base':
        return AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('distilroberta-base',model_max_length=args.max_length)
    elif args.model == 'char_cnn':
        from .char_cnn import CharacterLevelCNN, CharCNNTokenizer
        return CharacterLevelCNN(args), CharCNNTokenizer(args)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
    args.vocab_size = tokenizer.vocab_size
    if args.model == 'bert-base':
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=args.number_of_class)       
    elif args.model == 'word_cnn':
        from .word_cnn import WordLevelCNN
        model = WordLevelCNN(args)
    elif args.model == 'bilstm':
        from .lstm import LSTM
        model = LSTM(args)
    elif args.model == 'lstm':
        from .lstm import LSTM
        model = LSTM(args)
    elif args.model == 'rnn':
        from .rnn import RNN
        model = RNN(args)
    elif args.model == 'birnn':
        from .rnn import RNN
        model = RNN(args)
    else:
        raise Exception('Wrong model')
    return model, tokenizer