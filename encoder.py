import torch as t
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,num_layers=2 ,embedding_dim = 300,hidden_dim = 300, latent_variable_size = 300):
        super(Encoder, self).__init__()

      #  params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_variable_size = latent_variable_size
        self.num_layer = num_layers
      
        self.two_lstm = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.hidden_dim, num_layers = num_layers,
                                batch_first=True, bidirectional=False)
        
     
        #for reparametrization trick
        self.context_to_mu = nn.Linear(self.hidden_dim * 4, self.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.hidden_dim * 4, self.latent_variable_size)

    def forward(self, input, state = None):
        """
        :param sentences: [batch_size, seq_len, embed_size] tensor
        :param input_target: [batch_size, seq_len, embed_size] tensor
        :return: distributinon parameters of input sentenses with shape of 
            [batch_size, latent_variable_size]
        """

        
        [batch_size, seq_len, embed_size] = input.size()
        _, state = self.two_lstm(input, state)
        

        [h_state, c_state] = state
        h_state = h_state.permute(1,0,2).contiguous().view(batch_size, -1)
        c_state = c_state.permute(1,0,2).contiguous().view(batch_size, -1)
        final_state = t.cat([h_state, c_state], 1)
   
        mu, logvar = self.context_to_mu(final_state), self.context_to_logvar(final_state)
        return state, mu, logvar