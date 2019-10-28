
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, num_layers=2, embedding_dim = 300, hidden_dim = 300, latent_variable_size = 300):
        super(Discriminator, self).__init__()

    #  params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_variable_size = latent_variable_size
        self.num_layer = num_layers
        
        self.two_lstm = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.hidden_dim, num_layers = num_layers,
                                batch_first=True, bidirectional=False)
        
        self.linear = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()
        
     
    def forward(self, input):
        """
        :param input (sentences): [batch_size, seq_len, embed_size] tensor
        return final state: [[nb layer, batch, seq_len], [nb layer, batch, seq_len]]
        """
         
        state = None
        #[batch_size, seq_len, embed_size] = input.size()
        output, state = self.two_lstm(input, state)
       # input_linear = torch.cat((state[0][0],state[0][1]), dim  = 1)
        output_linear = self.linear(state[0][1])
        proba = self.sig(output_linear )
        return proba