import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: add code for transformer model that uses attention


class SelfAttnHead(nn.Module):
    def __init__(self, embed_dim, attn_head_size):
        super().__init__()

        # number of dimensions of token embedding
        self.embed_dim = embed_dim

        # number of dimensions of projection layer outputs as well as attention
        # head output
        self.attn_head_size = attn_head_size

        # lower triangular mask so tokens are not influenced by subsequent
        # tokens in the attention mechanism
        self.register_buffer('attn_mask', torch.tril(
            torch.ones(attn_head_size, attn_head_size)))

        # query, key, and value projection layers
        self.q_proj = nn.Linear(embed_dim, attn_head_size)
        self.k_proj = nn.Linear(embed_dim, attn_head_size)
        self.v_proj = nn.Linear(embed_dim, attn_head_size)

    def forward(self, x):
        # get query, key, and value projections
        # input: x
        # input shape: (batch size, context size, embed dim)
        # outputs: q, k, v
        # output shapes: (batch size, context size, attn head size),
        # (batch size, context size, attn head size),
        # (batch size, context size, attn head size)
        q = self.q_proj(x)
        k = self.q_proj(x)
        v = self.v_proj(x)

        # get attention patterns
        # inputs: q, k.mT
        # input shapes: (batch size, context size, attn head size),
        # (batch size, attn head size, context_size)
        # output: normalized attention pattern
        # output shape: (batch size, context size, context size)
        attn_pattern = q @ k.mT
        scaled_attn_pattern = attn_pattern / torch.sqrt(self.attn_head_size)
        masked_attn_pattern = scaled_attn_pattern * self.attn_mask
        # TODO: add masking based on where padding tokens are
        normalized_attn_pattern = torch.softmax(masked_attn_pattern, dim=1)

        # collect info across multiple token embeddings
        # inputs: normalized attn pattern, v
        # input shapes: (batch size, context size, context size),
        # (batch size, context size, attn head size)
        # output: information collected across
        # output shape: (batch size, context size, attn head size)
        collected_info = normalized_attn_pattern @ v
        return collected_info


class MultiHeadAttn(nn.Module):
    pass


class AttnBlock(nn.Module):
    pass


class PoetronModel(nn.Module):
    '''
    A foundation language model for generating/completing three-line
    poems
    '''
    def __init__(self, num_attn_heads=1, num_attn_blocks=1):
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.num_attn_blocks = num_attn_blocks

    def forward(self, x, y, tokenizer):
        # returns one-hot encoded next-token distributions from y
        # (TODO: replace with actual model that only accepts x as input)
        return F.one_hot(y, tokenizer.vocab_size)
