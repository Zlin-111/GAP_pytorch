
import torch

from torchtext import vocab
from torchtext.data import Field, BucketIterator, TabularDataset


def preprocess_text(path_data , path_to_train , 
                    path_to_val, path_to_test, path_to_glove, BATCH_SIZE, device):
 

    #function modified from http://anie.me/On-Torchtext/
    def init_emb(vocab, init="randn", num_special_toks=2):
        emb_vectors = vocab.vectors
        sweep_range = len(vocab)
        running_norm = 0.
        num_non_zero = 0
        total_words = 0
        for i in range(num_special_toks, sweep_range):
            if len(emb_vectors[i, :].nonzero()) == 0:
                # std = 0.05 is based on the norm of average GloVE 100-dim word vectors
                if init == "randn":
                    torch.nn.init.normal(emb_vectors[i], mean=0, std=0.05)
                    num_non_zero += 1 
                num_non_zero 
                running_norm += torch.norm(emb_vectors[i])
            total_words += 1
        print("number of unknown words are {}, total number of words are {}".format(
            num_non_zero, total_words))
        norm = running_norm / total_words
        emb_vectors / norm
    

  
    SRC = Field(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    TRG = Field(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    train_val_fields = [
        ('src', SRC ), 
        ('trg', TRG), 
    ]


    trainds, val, test = TabularDataset.splits(
        path=path_data , train= path_to_train,
        validation=path_to_val, test=path_to_test, format='csv',
        fields=train_val_fields)

    vec = vocab.Vectors(path_to_glove)
    SRC.build_vocab(trainds, val, test, vectors=vec)
    TRG.build_vocab(trainds, val, test,vectors=vec)

    init_emb(SRC.vocab, "randn",3)
    init_emb(TRG.vocab, "randn",3)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (trainds, val, test),sort_key=lambda x: len(x.src), batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE), device=device)
    
    return SRC, TRG, train_iter, val_iter, test_iter