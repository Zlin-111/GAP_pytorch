#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from prepro import preprocess_text
from  embeding_layer import  Embeding_layer
from  encoder import Encoder
from  highway import Highway
from  generator1 import Generator1
from generator2 import Generator2
from  discriminator import Discriminator

from train import train


def main():
    path_data = ''
    path_to_train = 'data_quora50k.csv'
    path_to_val = 'data_quora_val.csv'
    path_to_test = 'data_quora_test.csv'
    path_to_glove = 'glove.6B.50d.txt'
    
    BATCH_SIZE = 8
    embedding_dim = 50

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    SRC, TRG, train_iter, val_iter, test_iter = preprocess_text(path_data , path_to_train ,  path_to_val, path_to_test, path_to_glove, BATCH_SIZE, device)
    
    vocab_size_source = len(SRC.vocab)
    vocab_size_target = len(TRG.vocab)
    pretrained_vec_src = SRC.vocab.vectors
    pretrained_vec_trg = TRG.vocab.vectors
    print("vocab target size %s" % vocab_size_target)
    
    emb_s0 =  Embeding_layer(vocab_size_source, embedding_dim, pretrained_vec_src).to(device)
    emb_sp =  Embeding_layer(vocab_size_target, embedding_dim, pretrained_vec_trg).to(device)
    
    encoder1 = Encoder(num_layers=2 ,embedding_dim=embedding_dim, hidden_dim=300, latent_variable_size = 300).to(device)
    encoder2 = Encoder(num_layers=2 ,embedding_dim=embedding_dim, hidden_dim=300, latent_variable_size = 300).to(device)
    highway = Highway(size = embedding_dim, num_layers =2,  f=torch.nn.functional.relu).to(device)
    g1 = Generator1(num_layers=2 ,embedding_dim = embedding_dim, hidden_dim=300, latent_variable_size=300).to(device)
    g2 = Generator2(len(TRG.vocab),num_layers=2 ,embedding_dim = embedding_dim,hidden_dim=300, latent_variable_size = 300).to(device)
    
    discriminator = Discriminator( num_layers=2, embedding_dim = embedding_dim, hidden_dim=300, latent_variable_size=300).to(device)
    
    pad_idx = TRG.vocab.stoi['<pad>']
    
    lD_loss, models = train(5, train_iter, emb_s0, emb_sp, highway, encoder1, encoder2, g1, g2, discriminator, device, pad_idx)
    
    [highway, encoder1, encoder2, generator1, generator2, discriminator] =  models
    torch.save(highway, 'highway.pt')
    torch.save(encoder1, 'encoder1.pt')
    torch.save(encoder2, 'encoder2.pt')
    torch.save(generator2, 'generator2.pt')
    torch.save( generator1, 'generator1.pt')
    torch.save(discriminator, 'discriminator.pt')

if __name__ == '__main__':
    main()
    
    