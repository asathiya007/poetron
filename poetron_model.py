import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttnHead(nn.Module):
    def __init__(self, embed_dim, attn_head_size, context_size, input_mask):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        attn_head_size (int) - number of dimensions of projection layer outputs
        as well as attention head output
        context_size (int) - number of tokens in a single input
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_head_size = attn_head_size
        self.context_size = context_size
        self.input_mask = input_mask

        # lower triangular mask so tokens are not influenced by subsequent
        # tokens in the attention mechanism
        self.register_buffer('attn_mask', torch.tril(
            torch.ones(self.context_size, self.context_size)))

        # query, key, and value projection layers
        self.q_proj = nn.Linear(self.embed_dim, self.attn_head_size)
        self.k_proj = nn.Linear(self.embed_dim, self.attn_head_size)
        self.v_proj = nn.Linear(self.embed_dim, self.attn_head_size)

    def _get_init_attn_pattern(self, q, k):
        '''
        Input:
        q (torch.Tensor[float]) - tensor containing query vectors, shape is
        (batch size, context size, attn head size)
        k (torch.Tensor[float]) - tensor containing key vectors, shape is
        (batch size, context size, attn head size)

        Output:
        initial attention pattern (torch.Tensor[float]) - initial attention
        pattern, shape is (batch size, context size, context size)
        '''
        return q @ k.mT
    
    def _scale_attn_pattern(self, attn_pattern, scaling_factor):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        scaled attention pattern (torch.Tensor[float]) - scaled attention pattern,
        shape is (batch size, context size, context size)
        '''
        return attn_pattern * scaling_factor
    
    def _apply_subseq_mask(self, attn_pattern):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        masked attention pattern (torch.Tensor[float]) - attention pattern, with
        masking applied to subsequent tokens
        '''
        return attn_pattern.masked_fill(self.attn_mask == 0, float('-inf'))
    
    def _apply_input_mask(self, attn_pattern):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        masked attention pattern (torch.Tensor[float]) - attention pattern, with
        the input mask applied to each row (across the columns), shape is
        (batch size, context size, context size)
        '''
        return attn_pattern.masked_fill(
            # extra dimension added at dimension index 1 for broadcasting
            # each input mask across the rows of the corresponding batch input
            self.input_mask[:, None, :] == 0, float('-inf'))
    
    def _resolve_neg_inf_rows(self, attn_pattern):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        resolved attention pattern (torch.Tensor[float]) - attention pattern where
        all rows containing all -inf values are resolved, shape is (batch size,
        context size, context size)
        '''
        is_row_all_neginf = (attn_pattern == float('-inf')).all(dim=-1)
        batch_idxs = torch.nonzero(is_row_all_neginf)[:, 0]
        row_idxs = torch.nonzero(is_row_all_neginf)[:, 1]
        attn_pattern[batch_idxs, row_idxs, row_idxs] = 1
        return attn_pattern
    
    def _normalize_attn_pattern(self, attn_pattern):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        normalized attention pattern (torch.Tensor[float]), where all rows sum
        to 1
        '''
        return torch.softmax(attn_pattern, dim=-1)

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        self-attention head output (torch.Tensor[float]) - information collected
        across token embeddings to enrich them, shape is
        (batch size, context size, attn head size)
        '''

        # get query, key, and value projections of token embeddings
        # input: x (token embeddings)
        # input shape: (batch size, context size, embed dim)
        # outputs: q, k, v
        # output shapes: (batch size, context size, attn head size),
        # (batch size, context size, attn head size),
        # (batch size, context size, attn head size)
        q = self.q_proj(x)
        k = self.q_proj(x)
        v = self.v_proj(x)

        # get attention pattern
        init_attn_pattern = self._get_init_attn_pattern(q, k)
        # scaled attention pattern (scaled for numerical stability)
        scaled_attn_pattern = self._scale_attn_pattern(
            init_attn_pattern,
            1 / torch.sqrt(torch.Tensor([self.attn_head_size])).item())
        # masked attention pattern (masking applied to subsequent tokens)
        masked_attn_pattern = self._apply_subseq_mask(scaled_attn_pattern)
        # masked attention pattern (input mask applied)
        masked_attn_pattern = self._apply_input_mask(masked_attn_pattern)
        # if any row of the attention pattern is all -infinity, set the value
        # where the row and column indices are equal to 1, so those token
        # embeddings are enriched using only info that corresponds to that token
        # and not other tokens
        masked_attn_pattern = self._resolve_neg_inf_rows(masked_attn_pattern)
        # normalize attention pattern with softmax function, across rows
        normalized_attn_pattern = self._normalize_attn_pattern(
            masked_attn_pattern)

        # collect info across multiple token embeddings
        # inputs: normalized attn pattern, v
        # input shapes: (batch size, context size, context size),
        # (batch size, context size, attn head size)
        # output: information collected across
        # output shape: (batch size, context size, attn head size)
        collected_info = normalized_attn_pattern @ v
        return collected_info


class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim, context_size, input_mask, num_attn_heads,
                 attn_head_size):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)
        num_attn_heads (int) - number of self-attention heads
        attn_head_size (int) - number of dimensions of projection layer outputs
        as well as attention head output
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.input_mask = input_mask
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
    
        # create attention heads
        self.attn_heads = nn.ModuleList([
            SelfAttnHead(self.embed_dim, self.attn_head_size, self.context_size,
                         self.input_mask)
        ] * self.num_attn_heads)

        # output projection layer
        self.o_proj = nn.Linear(
            self.attn_head_size * self.num_attn_heads, self.embed_dim)

    def _get_concat_sah_outputs(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        concatenated self-attention head outputs (torch.Tensor[float]) - tensor
        of concatentated self-attention head outputs, shape is 
        (batch size, context size, attn head size * num attn heads)
        '''
        self_attn_head_outputs = [sah(x) for sah in self.attn_heads]
        concat_sah_outputs = torch.cat(self_attn_head_outputs, dim=-1)
        return concat_sah_outputs

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        multi-head attention output - information collected across token
        embeddings to enrich them, shape is
        (batch size, context size, embed dim)
        '''

        # get information to enrich token embeddings from each self-attention
        # head
        concat_sah_outputs = self._get_concat_sah_outputs(x)

        # project the concatenated self-attention head outputs to the embedding
        # space to get the multi-head attention output (information to enrich
        # the token embeddings)
        # input: cocnatenated self-attention head outputs
        # input shape: (batch size, context size, attn head size * num attn
        # heads)
        # output: multi-head attention output
        # output shape: (batch size, context size, embed dim)
        multi_head_attn_output = self.o_proj(concat_sah_outputs) 

        # return multi-head attention output
        return multi_head_attn_output


class FeedFwd(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_hidden_layers):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        hidden_size (int) - hidden size of the feed-forward network
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # define layers of network
        input_layer = [
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU()
        ]
        hidden_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ] * num_hidden_layers
        output_layer = [
            nn.Linear(hidden_size, embed_dim)
        ]
        self.layers = nn.Sequential(
            *(input_layer + hidden_layers + output_layer))
    
    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of enriched token embeddings, shape is
        (batch_size, context size, embed dim)

        Output:
        feed-forward network output (torch.Tensor[float]) - tensor containing
        information to further enrich token embeddings, shape is
        (batch size, context size, embed dim)
        '''
        return self.layers(x)


class AttnBlock(nn.Module):
    def __init__(self, embed_dim, context_size, input_mask, num_attn_heads,
                 attn_head_size, hidden_size, num_hidden_layers):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)
        num_attn_heads (int) - number of self-attention heads
        attn_head_size (int) - number of dimensions of attention projection
        layer outputs as well as attention head output
        hidden_size (int) - hidden size of the feed-forward network
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.input_mask = input_mask
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # multi-head attention
        self.mah = MultiHeadAttn(
            self.embed_dim, self.context_size, self.input_mask,
            self.num_attn_heads, self.attn_head_size)

        # layer normalization
        self.post_mah_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ffwd_layer_norm = nn.LayerNorm(self.embed_dim)

        # feed forward network
        self.ffwd = FeedFwd(
            self.embed_dim, self.hidden_size, self.num_hidden_layers)

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        attention block output (torch.Tensor[float]) - tensor of enriched token
        embeddings, shape is (batch size, context size, embed dim)
        '''
        # enrich token embeddings with multi-head attention
        x += self.mah(x)

        # layer normalization
        x = self.post_mah_layer_norm(x)

        # enrich token embeddings with feed forward network
        x += self.ffwd(x)

        # layer normalization
        x = self.post_ffwd_layer_norm(x)

        # return enriched token embeddings
        return x


class PoetronModel(nn.Module):
    '''
    A language model for generating/completing three-line poems
    '''
    def __init__(self, vocab_size, embed_dim, context_size, input_mask,
                 num_attn_heads, attn_head_size, hidden_size, num_hidden_layers,
                 num_attn_blocks):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)
        num_attn_heads (int) - number of self-attention heads per attention
        block
        attn_head_size (int) - number of dimensions of attention projection
        layer outputs as well as attention head output
        hidden_size (int) - hidden size of the feed-forward network in each
        attention block
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network of each attention block
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.input_mask = input_mask
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_blocks = num_attn_blocks

        # input token embedding matrix
        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # TODO: sinusoidal positional embedding

        # attention blocks
        self.attn_blocks = nn.Sequential(*[
            AttnBlock(
                self.embed_dim, self.context_size, self.input_mask,
                self.num_attn_heads, self.attn_head_size, self.hidden_size,
                self.num_hidden_layers)
        ] * self.num_attn_blocks)

        # output projection layer
        self.o_proj = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of input tokens, shape is
        (batch size, context size)

        Output:
        transformer model output (torch.Tensor[float]) - tensor of logits
        (enriched token embeddings, projected from embed_dim dimensions
        to vocab_size dimensions), shape is (batch size, context size,
        vocab size)
        '''

        # get input token embeddings
        input_tok_embeds = self.input_embed(x)

        # TODO: get positional embeddings and enrich token embeddings with
        # positional info

        # enrich token embeddings through attention blocks
        enriched_tok_embeds = self.attn_blocks(input_tok_embeds)

        # project token embeddings from embed_dim dimensions to vocab_size
        # dimensions to get logits
        logits = self.o_proj(enriched_tok_embeds)

        # return logits
        return logits
        
