import torch
import torch.nn as nn
import dataloader
from get_device import device

num_char = dataloader.total_char
num_cond = dataloader.total_tense

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, latent_size, tense_size=4):
        super(EncoderRNN, self).__init__()
        self.latent_size = latent_size
        self.tense_size = tense_size

        self.embedding_in = nn.Embedding(num_char, latent_size)
        self.embedding_cond = nn.Embedding(num_cond, tense_size)
        #  add hidden size for  concatenated conditioning tense value
        self.lstm = nn.LSTM(latent_size, latent_size*2 + tense_size)
        self.fc_mu_h = nn.Linear(latent_size*2 + tense_size, latent_size)
        self.fc_var_h = nn.Linear(latent_size*2 + tense_size, latent_size)
        self.fc_mu_c = nn.Linear(latent_size*2 + tense_size, latent_size)
        self.fc_var_c = nn.Linear(latent_size*2 + tense_size, latent_size)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu
        #return mu

    def forward(self, mode, input, hidden, cell, condition):
        # `mode` can be either 'train' or 'eval'
        # `input` should have shape (seq_len, batch_size , input_dim)
        # which is  (seq_len , 1, 1)

        # 'hidden' and  'cell' should have shape (1, 1, hidden_size*2)
        # ,for encodeing mu and variance)

        # 'condition' should have shape (1,1,1)
        seq_len = input.shape[0]
        embedded_input = self.embedding_in(input)
        embedded_input = embedded_input.view(seq_len, 1, self.latent_size)

        embedded_tense = self.embedding_cond(condition)
        embedded_tense = embedded_tense.view(1, 1, self.tense_size)

        hidden_cond  = torch.cat([hidden, embedded_tense], -1)
        cell_cond  = torch.cat([cell, embedded_tense], -1)

        output, (hidden_n, cell_n) = self.lstm(embedded_input,
                                               (hidden_cond, cell_cond))
        output = output[-1, :, :]
        mu_h = self.fc_mu_h(output)
        log_var_h = self.fc_var_h(output)

        mu_c = self.fc_mu_c(output)
        log_var_c = self.fc_var_c(output)

        zh = self.reparametrize(mu_h, log_var_h)
        zc = self.reparametrize(mu_c, log_var_c)

        if mode == 'train':
            kl_loss_h = self.regularization_loss(mu_h, log_var_h)
            kl_loss_c = self.regularization_loss(mu_c, log_var_c)
            kl_loss = kl_loss_h + kl_loss_c
            return kl_loss, zh, zc
        else:
            assert mode == 'eval', 'unknown mode for encoder'
            return zh, zc
    def regularization_loss(self, mu, log_var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kl_loss

    def initHidden(self):
        return torch.zeros(1, 1, self.latent_size*2, device=device)
    def initCell(self):
        return torch.zeros(1, 1, self.latent_size*2, device=device)



# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, latent_size,  tense_size=4):
        super(DecoderRNN, self).__init__()
        self.latent_size = latent_size
        self.tense_size = tense_size

        self.embedding_in = nn.Embedding(num_char, num_char)

        self.lstm = nn.LSTM(num_char, latent_size + tense_size)

        self.out = nn.Sequential(
            nn.Linear(latent_size + tense_size, num_char),
            nn.LogSoftmax(dim=1)
        )

        self.embedding_cond = nn.Embedding(num_cond, tense_size)
        self.criterion = nn.NLLLoss()
        self.decoder_input0 = torch.tensor([[[26]]], device=device)
    def one_step(self, input, hidden, cell):
        # input should have shape (1, 1, 1)
        # hidden, cell both should have shape (1, 1, latent_size + tense_size)
        embedded_input = self.embedding_in(input)
        embedded_input = embedded_input.view(1, 1, -1)

        output,  (hidden_n, cell_n) = self.lstm(embedded_input, (hidden, cell))
        # now output.shape = (1, 1, latent_size + tense_size)

        output = output.view(1, -1)
        output = self.out(output)
        # now output.shape = (1, num_char)

        return output, hidden_n , cell_n

    def forward(self, mode,  hidden, cell, condition, target=None,
                use_teacher_forcing=True):
        # 'mode' should be either 'train' or 'eval'
        # In 'train' mode, 'target' must be provided,
        # and only 'rec_loss' is returned
        # In 'eval' mode, a generated code sequence is returned (excluding EOS)


        # hidden, cell  should have shape (1, 1, latent_size)
        # condition should have shape (1, 1, 1)
        # target should have shape (seq_len, 1, 1)
        # and target should be append with EOS token ('$')

        # print(condition)

        embedded_cond = self.embedding_cond(condition)
        embedded_cond = embedded_cond.view(1, 1, -1)

        hidden_cond = torch.cat([hidden, embedded_cond], -1)
        cell_cond = torch.cat([cell, embedded_cond], -1)

        if mode == 'train':
            assert target is not None, 'in train mode, target must be provided'
            target_length = target.shape[0]
            # start of sequence
            decoder_input = self.decoder_input0

            h = hidden_cond
            c = cell_cond
            rc_loss = 0 # reconstruction loss

            if use_teacher_forcing:
                for di in range(target_length):
                    decoder_output, h, c = self.one_step(decoder_input, h, c)

                    rc_loss += self.criterion(decoder_output,
                                              target[di].view(1,))
                    decoder_input = target[di, :, :]
            else:
                for di in range(target_length):
                    decoder_output, h, c = self.one_step(decoder_input,
                                                         h, c)

                    rc_loss += self.criterion(decoder_output,
                                              target[di].view(1))
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach().view(1, 1, 1)
            rc_loss /= target_length
            return rc_loss
        else:
            assert mode == 'eval', "unknown mode for decoder"
            max_length = 20
            # start of sequence
            decoder_input = self.decoder_input0

            h = hidden_cond
            c = cell_cond
            result = []
            for i in range(max_length):

                decoder_output, h, c = self.one_step(decoder_input, h, c)
                topv, topi = decoder_output.topk(1)


                # print(decoder_output)
                # print(topi)
                # print(topi.item())
                char_code = topi.detach().cpu()

                if char_code == 27:  # End of Sequence
                    break
                result.append(char_code)
                decoder_input = topi.detach().view(1, 1, 1)

            return result

    def initHidden(self):
        return torch.zeros(1, 1, self.latent_size, device=device)
    def initCell(self):
        return torch.zeros(1, 1, self.latent_size, device=device)

if __name__ == '__main__':
    from torch.utils import data
    from dataloader import get_training_pairs
    dl_train = data.DataLoader(get_training_pairs(),batch_size=1, shuffle=True)

    latent_size = 128
    enc = EncoderRNN(latent_size).to(device)
    dec = DecoderRNN(latent_size).to(device)
    for w, t in dl_train:
        w = w.transpose(0,1).to(device)
        t = t.to(device)
        h0 = enc.initHidden()
        c0 = enc.initCell()
        kl_loss, code = enc('train', w, h0, c0, t)
        print(kl_loss)
        h0 = dec.initHidden()
        c0 = code.view(1, 1, -1)
        rc_loss = dec('train', h0, c0, t, w)
        print(rc_loss)
        break
