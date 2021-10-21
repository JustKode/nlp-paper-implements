# reference: https://github.com/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, start_vocab_size, end_vocab_size, batch_size, n_hidden):
        super(Attention, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = n_hidden
        self.start_vocab_size = start_vocab_size
        self.end_vocab_size = end_vocab_size

        self.attn = nn.Linear(n_hidden, n_hidden)  # for attention function
        self.out1 = nn.Linear(n_hidden * 2, n_hidden)
        self.out2 = nn.Linear(n_hidden, end_vocab_size)
        self.enc_cell = nn.LSTM(input_size=start_vocab_size, hidden_size=n_hidden, dropout=0.3)  # [start_step, batch_size, start_vocab_size]
        self.dec_cell = nn.LSTM(input_size=end_vocab_size, hidden_size=n_hidden, dropout=0.3)  # [end_step, batch_size, end_vocab_size]

    def forward(self, input_batch, output_batch):
        batch_size = len(input_batch)
        input_batch = input_batch.transpose(0, 1)  # [input_step, batch_size]
        output_batch = output_batch.transpose(0, 1)  # [output_step, batch_size]

        start_step = len(input_batch)  # input_step
        end_step = len(output_batch)  # output_step

        enc_inputs = F.one_hot(input_batch, num_classes=self.start_vocab_size).float()  # [input_step, batch_size, input_vocab_size]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs)  # enc_outputs: [input_step, batch_size, hidden_size]

        hidden = enc_hidden
        dec_inputs = F.one_hot(output_batch, num_classes=self.end_vocab_size).float()  # [output_step, batch_size, end_vocab_size]

        model = torch.empty([end_step, batch_size, self.end_vocab_size])  # [output_step, batch_size, end_vocab_size]

        for i in range(end_step):
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)  # dec_output: [1, batch_size, hidden_size]
            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # [batch_size, start_step]
            context = attn_weights.view(-1, 1, start_step).bmm(enc_outputs.transpose(0, 1))  # [batch_size, 1, start_step] x [batch_size, start_step, hidden_size]
            dec_output = dec_output.squeeze(0)  # [batch_size, hidden_size]
            context = context.squeeze(1)  # [batch_size, hidden_size]
            model[i] = self.out2(torch.tanh(self.out1(torch.cat((dec_output, context), 1)))) # [batch_size, hidden_size * 2] @ [hidden_size * 2, embed_size]

        return F.softmax(model.transpose(0, 1), dim=2)  # [batch_size, output_step, embed_size] => [batch_size, output_step]

    def get_att_weight(self, dec_output, enc_outputs):
        # enc_outputs: [input_step, batch_size, hidden_size]
        # dec_output: [1, batch_size, hidden_size]

        input_step = len(enc_outputs)
        attn_scores = torch.zeros(input_step, self.batch_size)

        for i in range(input_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])  # [batch_size]

        attn_scores = attn_scores.transpose(0, 1)  # [batch_size, input_step]
        return F.softmax(attn_scores, dim=1)  # [batch_size, input_step]

    def get_att_score(self, dec_output, enc_output):
        # enc_output: [batch_size, hidden_size]
        # dec_output: [1, batch_size, hidden_size]
        score = self.attn(enc_output)
        return torch.sum(score * dec_output.view(self.batch_size, self.hidden_size), dim=-1)

    def get_word(self, output):
        output = F.log_softmax(self.word(output), dim=2)
        return output.argmax(dim=2)
