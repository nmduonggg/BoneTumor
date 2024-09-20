from functools import partial
import math
import os
from typing import Iterable, List

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 16*16):  # 2^12 = 16* 2^8
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):  # vocab size is the number of classes 
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
# Seq2Seq Network
class TransformerReorder(nn.Module):
    def __init__(self,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 emb_size: int = 1024,
                 nhead: int = 8,
                 vocab_size: int = 16*16,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor, ):
        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb)   # mask is skipped 
        return self.generator(outs)

    def encode(self, src: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)))

    def decode(self, tgt: Tensor, memory: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                                        self.tgt_tok_emb(tgt)), memory, )