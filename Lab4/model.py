import torch
import torch.nn as nn



#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, tense_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tense_size = tense_size

        self.embedding_in = nn.Embedding(input_size, hidden_size)
        self.embedding_cond = nn.Embedding(1,tense_size)
        #  add hidden size for  concatenated conditioning tense value
        self.lstm = nn.LSTM(hidden_size, hidden_size*2 + tense_size)
        self.fc_mu = nn.Linear(hidden_size*2 + tense_size, hidden_size)
        self.fc_var = nn.Linear(hidden_size*2 + tense_size, hidden_size)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def forward(self, input, hidden, cell, condition):
        # `input` should have shape (seq_len, batch_size , input_dim)
        # which is  (seq_len , 1, 1)

        # 'hidden' and  'cell' should have shape (1, 1, hidden_size*2)
        # ,for encodeing mu and variance)

        # 'condition' should have shape (1,1,1)
        seq_len = input.shape[0]
        embedded_input = self.embedding_in(input)
        embedded_input = embedded_input.view(seq_len, 1, self.hidden_size)

        embedded_tense = self.embedding_cond(condition)
        embedded_tense = embedded_tense.view(1, 1, self.tense_size)

        hidden_cond  = torch.cat([hidden, embedded_tense], -1)
        cell_cond  = torch.cat([cell, embedded_tense], -1)
        output, hidden_n , cell_n = self.lstm(embedded_input,
                                              hidden_cond,
                                              cell_cond)
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size*2, device=device)
    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size*2, device=device)



#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, ):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
