
import torch.nn as nn


class Generator1(nn.Module):
    def __init__(self,num_layers=2 ,embedding_dim = 50,hidden_dim = 300, latent_variable_size = 600):
        super(Generator1, self).__init__()

      #  params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_variable_size = latent_variable_size
        self.num_layer = num_layers
      
        self.two_lstm = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.hidden_dim, num_layers = num_layers,
                                batch_first=True, bidirectional=False)
        
     
    def forward(self, input):
        """
        :param input (sentences): [batch_size, seq_len, embed_size] tensor
        return final state: [[nb layer, batch, seq_len], [nb layer, batch, seq_len]]
        """

        state = None
        _, state = self.two_lstm(input, state)
        return state