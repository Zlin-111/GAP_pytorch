

import torch.nn as nn

class Embeding_layer(nn.Module): 
    
    def __init__(self, vocab_size, embedding_dim, pretrained_vec):
        """
         
         return a  input_source: [batch_size, seq_len, embed_size] tensor"""
        super(Embeding_layer, self).__init__()
            
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
            
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec) # load pretrained vectors
        self.emb.weight.requires_grad = False # make embedding non trainable
        
    def forward(self, sentence): #hidden = (h0, cself.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        """sentence dimension : [seq_len, batch_size] 
           hidden   dimension  : """
        
        embedded_sentence = self.emb(sentence).permute(1,0,2)
        return embedded_sentence
