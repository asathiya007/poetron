import torch
import torch.nn as nn
import math


class SelfAttnHead(nn.Module):
    def __init__(self, embed_dim, attn_head_size, context_size):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        attn_head_size (int) - number of dimensions of projection layer outputs
        as well as attention head output
        context_size (int) - number of tokens in a single input
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_head_size = attn_head_size
        self.context_size = context_size

        # lower triangular mask so tokens are not influenced by subsequent
        # tokens in the attention mechanism
        self.register_buffer('causal_mask', torch.tril(
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
    
    def _apply_causal_mask(self, attn_pattern):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        masked attention pattern (torch.Tensor[float]) - attention pattern, with
        masking applied to subsequent tokens
        '''
        return attn_pattern.masked_fill(self.causal_mask == 0, float('-inf'))
    
    def _apply_input_mask(self, attn_pattern, input_mask):
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
            input_mask[:, None, :] == 0, float('-inf'))
    
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

    def forward(self, x, input_mask):
        '''
        Input:
        x (torch.Tensor[float]) - token embeddings, shape is
        (batch size, context size, embed dim)
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)

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
        masked_attn_pattern = self._apply_causal_mask(scaled_attn_pattern)
        # masked attention pattern (input mask applied)
        masked_attn_pattern = self._apply_input_mask(
            masked_attn_pattern, input_mask)
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
    def __init__(self, embed_dim, context_size, num_attn_heads, attn_head_size):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        num_attn_heads (int) - number of self-attention heads
        attn_head_size (int) - number of dimensions of projection layer outputs
        as well as attention head output
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
    
        # create attention heads
        self.attn_heads = nn.ModuleList([])
        for _ in range(self.num_attn_heads):
            self.attn_heads.append(SelfAttnHead(
                self.embed_dim, self.attn_head_size, self.context_size))

        # output projection layer
        self.o_proj = nn.Linear(
            self.attn_head_size * self.num_attn_heads, self.embed_dim)

    def _get_concat_sah_outputs(self, x, input_mask):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)

        Output:
        concatenated self-attention head outputs (torch.Tensor[float]) - tensor
        of concatentated self-attention head outputs, shape is 
        (batch size, context size, attn head size * num attn heads)
        '''
        self_attn_head_outputs = [sah(x, input_mask) for sah in self.attn_heads]
        concat_sah_outputs = torch.cat(self_attn_head_outputs, dim=-1)
        return concat_sah_outputs

    def forward(self, x, input_mask):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)

        Output:
        multi-head attention output - information collected across token
        embeddings to enrich them, shape is
        (batch size, context size, embed dim)
        '''

        # get information to enrich token embeddings from each self-attention
        # head
        concat_sah_outputs = self._get_concat_sah_outputs(x, input_mask)

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
        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])
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
    def __init__(self, embed_dim, context_size, num_attn_heads, attn_head_size,
                 hidden_size, num_hidden_layers):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
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
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # multi-head attention
        self.mah = MultiHeadAttn(
            self.embed_dim, self.context_size, self.num_attn_heads,
            self.attn_head_size)

        # layer normalization
        self.post_mah_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ffwd_layer_norm = nn.LayerNorm(self.embed_dim)

        # feed forward network
        self.ffwd = FeedFwd(
            self.embed_dim, self.hidden_size, self.num_hidden_layers)

    def forward(self, x, input_mask):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)

        Output:
        attention block output (torch.Tensor[float]) - tensor of enriched token
        embeddings, shape is (batch size, context size, embed dim)
        '''
        # enrich token embeddings with multi-head attention
        x = x + self.mah(x, input_mask)

        # layer normalization
        x = self.post_mah_layer_norm(x)

        # enrich token embeddings with feed forward network
        x = x + self.ffwd(x)

        # layer normalization
        x = self.post_ffwd_layer_norm(x)

        # return enriched token embeddings
        return x


class PoetronModel(nn.Module):
    '''
    A language model for generating/completing short poems
    '''
    def __init__(self, vocab_size, embed_dim, context_size, num_attn_heads,
                 attn_head_size, hidden_size, num_hidden_layers,
                 num_attn_blocks, device):
        '''
        Input:
        vocab_size (int) - number of tokens in the vocabulary
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        num_attn_heads (int) - number of self-attention heads per attention
        block
        attn_head_size (int) - number of dimensions of attention projection
        layer outputs as well as attention head output
        hidden_size (int) - hidden size of the feed-forward network in each
        attention block
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network of each attention block
        num_attn_blocks (int) - number of attention blocks
        device (torch.device) - the device that the model is running on
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_blocks = num_attn_blocks
        self.device = device

        # input token embedding matrix
        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # sinusoidal positional embedding
        self.sin_pos_embeds = self._get_sin_pos_embeds()

        # attention blocks
        self.attn_blocks = nn.ModuleList([])
        for _ in range(self.num_attn_blocks):
            self.attn_blocks.append(AttnBlock(
                self.embed_dim, self.context_size, self.num_attn_heads,
                self.attn_head_size, self.hidden_size, self.num_hidden_layers))

        # output projection layer
        self.o_proj = nn.Linear(self.embed_dim, self.vocab_size)

    def _get_sin_pos_embeds(self):
        '''
        Computes sinusoidal positional embeddings.
        
        Input:
        None

        Output:
        sinusoidal positional embeddings (torch.Tensor[float]) - tensor of
        sinusoidal positional embeddings, shape is (context size,
        embed_dim)
        '''

        # get positions vector, shape is (context size, )
        positions = torch.arange(self.context_size)

        # compute frequencies (log is used to avoid overflow errors, when
        # raising 10000 to a large power) across all positions, shape is
        # (context size, half embed dim)
        half_embed_dim = math.ceil(self.embed_dim / 2)
        frequencies = positions[:, None] * torch.exp(
            -1 * (2 / self.embed_dim) * torch.arange(half_embed_dim)
            * math.log(10000))[None, :]

        # get sin and cos values, interleave them (by stacking along a new
        # dimension and reshaping), and truncate at original embed dim
        # to get positional embeddings of shape (context size, embed dim)
        sin_values = frequencies.sin()
        cos_values = frequencies.cos()
        sin_pos_embeds = torch.stack([sin_values, cos_values], dim=-1)\
            .reshape(self.context_size, half_embed_dim * 2)
        sin_pos_embeds = sin_pos_embeds[:, :self.embed_dim]

        # return sinusoidal positional embeddings
        return sin_pos_embeds

    def _get_pos_embeds(self, input_mask):
        '''
        Input:
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)
        
        Output:
        positional embeddings (torch.Tensor[int]) - tensor of positional
        embeddings (zero vectors for padding tokens at the start of the
        sequence, sinusoidal positional embeddings for the rest of the tokens)
        '''

        # generate positional embeddings, using sinusoidal positional embedding
        # (0s for padding tokens at the beginning of the sequence)
        first_nonzero_idxs = torch.argmax(input_mask, dim=1).flatten().tolist()
        pos_embeds = []
        for i in range(len(first_nonzero_idxs)):
            # do not include any positional info for masks of all 0s
            if torch.equal(input_mask[i], torch.zeros(
                    self.context_size, device=self.device)):
                pos_embeds.append(
                    torch.zeros(self.context_size, self.embed_dim))
            # only include positional info after padding tokens at the start of
            # the sequence
            else:
                num_starting_pad_toks = first_nonzero_idxs[i]
                pos_embed = torch.cat([
                    torch.zeros(num_starting_pad_toks, self.embed_dim),
                    self.sin_pos_embeds[
                        :self.context_size - num_starting_pad_toks]
                ], dim=0)
                pos_embeds.append(pos_embed)
        pos_embeds = torch.stack(pos_embeds, dim=0)
        return pos_embeds

    def forward(self, x, input_mask):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of input tokens, shape is
        (batch size, context size)
        input_mask (torch.Tensor[int]) - mask on input tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (batch size, context size)

        Output:
        transformer model output (torch.Tensor[float]) - tensor of logits
        (enriched token embeddings, projected from embed_dim dimensions
        to vocab_size dimensions), shape is (batch size, context size,
        vocab size)
        '''

        # get input token embeddings
        input_tok_embeds = self.input_embed(x)

        # enrich token embeddings with positional info
        pos_embeds = self._get_pos_embeds(input_mask).to(self.device)
        enriched_tok_embeds = input_tok_embeds + pos_embeds

        # enrich token embeddings through attention blocks
        for attn_block in self.attn_blocks:
            enriched_tok_embeds = attn_block(enriched_tok_embeds, input_mask)

        # project token embeddings from embed_dim dimensions to vocab_size
        # dimensions to get logits
        logits = self.o_proj(enriched_tok_embeds)

        # return logits
        return logits
