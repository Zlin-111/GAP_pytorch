import torch
import torch.nn as nn

class Generator2(nn.Module):
    def __init__(self,output_dim,num_layers=2 ,embedding_dim = 300, hidden_dim = 300, latent_variable_size = 300):
        super(Generator2, self).__init__()

      #  params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_variable_size = latent_variable_size
        self.num_layer = num_layers
      
        self.two_lstm = nn.LSTM(input_size = self.embedding_dim + self.latent_variable_size,
                                hidden_size = self.hidden_dim, num_layers = num_layers,
                                batch_first=True, bidirectional=False)
        
        self.out = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim = 2)
     
    def forward(self, input, z, state):
        input = input.unsqueeze(1)
        z = z.unsqueeze(1)
        input = torch.cat((input,z), dim=2)

        output, state = self.two_lstm(input, state)
        result = self.out(output)
        result =  self.log_softmax(result) 
        world = result.argmax(dim = 2)
        return world, result, state  
      
    #https://github.com/suragnair/seqGAN/blob/master/generator.py
    #https://arxiv.org/pdf/1908.05551.pdf
