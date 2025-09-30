import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Linear
import pdb

from utils import get_similarity
from .functions import get_mask, refer_points


class PositionalEmbeddingLayer(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEmbeddingLayer, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] + x


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
            ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):

        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs)

        return all_inputs


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        if mask is not None:
            try:
                attn = attn.masked_fill(mask, float('-inf'))
            except:
                mask = mask.byte()
                attn = attn.masked_fill(mask, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, opt, d_model, d_inner, n_head, d_k, d_v, inner_size, dropout=0.1,
                 normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.opt = opt
        self.slf_attn_temporal = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                                    normalize_before=normalize_before)
        self.slf_attn_semantic = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                                    normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)
        self.inner_size = inner_size


    def get_semantic_mask(self, enc_input, all_size):
        levels = list(range(self.opt.level))
        similarity = [get_similarity(enc_input[i].unsqueeze(0), enc_input[i].unsqueeze(0)).squeeze(0) for i in
                      range(enc_input.shape[0])]
        similarity = torch.stack(similarity, dim=0).squeeze(1)
        semantic_mask = torch.zeros_like(similarity, dtype=torch.bool)
        for batch in range(enc_input.shape[0]):
            column_index = 0
            for level in range(len(all_size)):
                row_slice = slice(column_index, (column_index + all_size[level]), 1)
                row_slice_ = torch.arange(0, all_size[level]).unsqueeze(-1)
                if level in levels:  # the bottom layer
                    # this layer
                    column_slice = slice(column_index, (column_index + all_size[level]), 1)
                    s = similarity[batch][row_slice, column_slice]
                    indices = s.topk(min(all_size[level], self.inner_size))[1]
                    semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True
                    # upper layer
                    # column_slice = slice((column_index + self.all_size[level]),
                    #                      (column_index + self.all_size[level] + self.all_size[level + 1]), 1)
                    # s = similarity[batch][row_slice, column_slice]
                    # indices = s.topk(1)[1]
                    # semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True

                    # update column_index
                    column_index += all_size[level]
                # elif level == len(self.all_size) - 1:
                #     # this layer
                #     column_slice = slice(column_index, (column_index + self.all_size[level]), 1)
                #     s = similarity[batch][row_slice, column_slice]
                #     indices = s.topk(min(self.all_size[level], self.inner_size))[1]
                #     semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True
                #
                #     # lower layer
                #     column_slice = slice((column_index - self.all_size[level - 1]), column_index, 1)
                #     s = similarity[batch][row_slice, column_slice]
                #     indices = s.topk(self.inner_size)[1]
                #     semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True
                #
                #     # update column_index
                #     column_index += self.all_size[level]
                # else:
                #     # this layer
                #     column_slice = slice(column_index, (column_index + self.all_size[level]), 1)
                #     s = similarity[batch][row_slice, column_slice]
                #     indices = s.topk(min(self.all_size[level], self.inner_size))[1]
                #     semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True
                #     # upper layer
                #     column_slice = slice((column_index + self.all_size[level]),
                #                          (column_index + self.all_size[level] + self.all_size[level + 1]), 1)
                #     s = similarity[batch][row_slice, column_slice]
                #     indices = s.topk(1)[1]
                #     semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True
                #     # lower layer
                #     column_slice = slice((column_index - self.all_size[level - 1]), column_index, 1)
                #     s = similarity[batch][row_slice, column_slice]
                #     indices = s.topk(self.inner_size)[1]
                #     semantic_mask[batch][row_slice, column_slice][row_slice_, indices] = True
                #     # update column_index
                #     column_index += self.all_size[level]

        return semantic_mask

    def forward(self, enc_input, all_size, slf_attn_mask=None):
        residual = enc_input
        enc_output_temporal, enc_slf_attn = self.slf_attn_temporal(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output_semantic, enc_slf_attn = self.slf_attn_semantic(enc_input, enc_input, enc_input,
                                                                   mask=self.get_semantic_mask(enc_input, all_size))
        enc_output = (enc_output_temporal + enc_output_semantic + residual) / 3.0
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        self.all_size = None
        self.layers = nn.ModuleList([
            EncoderLayer(opt, opt.d_model, opt.d_inner, opt.n_head, opt.d_k, opt.d_v, opt.semantic_node,
                         dropout=opt.dropout,
                         normalize_before=False) for i in range(opt.n_layer)
        ])

        self.conv_layers = Bottleneck_Construct(opt.d_model, opt.window_size, opt.d_inner)

        # 定长版
        self.mask, self.all_size = self.get_mask(opt.input_size, self.opt.window_size, self.opt.inner_size,
                                       self.opt.device)
        self.indexes = refer_points(self.all_size, self.opt.window_size, self.opt.device)

    def get_mask(self, input_size, window_size, inner_size, device):
        """Get the attention mask of Pyraformer-Naive"""
        # Get the size of all layers
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)

        seq_length = sum(all_size)
        mask = torch.zeros(seq_length, seq_length, device=device)

        # get intra-scale mask
        inner_window = inner_size // 2
        for layer_idx in range(len(all_size)):
            start = sum(all_size[:layer_idx])
            for i in range(start, start + all_size[layer_idx]):
                left_side = max(i - inner_window, start)
                right_side = min(i + inner_window + 1, start + all_size[layer_idx])
                mask[i, left_side:right_side] = 1

        # get inter-scale mask
        for layer_idx in range(1, len(all_size)):
            start = sum(all_size[:layer_idx])
            for i in range(start, start + all_size[layer_idx]):
                left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
                if i == (start + all_size[layer_idx] - 1):
                    right_side = start
                else:
                    right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
                mask[i, left_side:right_side] = 1
                mask[left_side:right_side, i] = 1

        mask = (1 - mask).to(torch.bool)

        return mask, all_size

    def forward(self, x_enc):
        seq_enc = x_enc  # embedding

        self.mask, self.all_size = self.get_mask(x_enc.shape[1], self.opt.window_size, self.opt.inner_size,
                                                 self.opt.device)
        self.indexes = refer_points(self.all_size, self.opt.window_size, self.opt.device)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)
        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, self.all_size, mask)
        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return seq_enc


class MLP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.d_model = opt.d_model
        self.l = len(opt.window_size) + 1
        self.input_size = opt.inner_size
        self.hidden_dim = opt.MLP_hidden
        self.LinearLayer1 = nn.Linear(self.d_model * self.l, self.hidden_dim)
        nn.init.xavier_uniform_(self.LinearLayer1.weight)
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.LinearLayer2 = nn.Linear(self.hidden_dim, opt.output_size)
        nn.init.xavier_uniform_(self.LinearLayer2.weight)

    def forward(self, x):
        x = self.LinearLayer1(x)
        x = self.LayerNorm(x)
        x = self.relu(x)
        x = self.LinearLayer2(x)

        return x
