import kagglehub
import os
import pandas as pd
from poetron_model import PoetronModel
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


# set integer data type
INT_DATA_TYPE = torch.int32


class PoetronTokenizer:
    '''
    Tokenizer for the Poetron language model.
    '''
    def __init__(self, special_tokens, padding_token, token_to_int,
                 int_to_token, max_length, non_alnum_chars):
        self.special_tokens = special_tokens
        self.padding_token = padding_token
        self.token_to_int = token_to_int
        self.int_to_token = int_to_token
        self.vocab_size = len(self.token_to_int.keys())
        self.max_length = max_length
        self.non_alnum_chars = non_alnum_chars

    def pad_or_truncate(self, tensor):
        '''
        Input:
        tensor (torch.Tensor[int]) - a sequence of tokens to pad/truncate

        Output:
        new_tensor (torch.Tensor[int]) - the sequence of tokens that has
        either been padded (if length < self.max_length) or truncated (if
        length > self.max_length), with shape (self.max_length)
        mask (torch.Tensor[int]) - mask on sequence of tokens (0 for tokens to
        ignore (like padding tokens), 1 for tokens to collect info from), shape
        is (self.max_length)
        '''
        if len(tensor) < self.max_length:
            new_tensor = torch.cat(
                [torch.ones(self.max_length - len(tensor))
                    * self.token_to_int[self.padding_token], tensor],
                dim=-1)
        elif len(tensor) > self.max_length:
            new_tensor = tensor[-self.max_length:]
        new_tensor = new_tensor.type(INT_DATA_TYPE)
        mask = (new_tensor != self.token_to_int[self.padding_token]).type(
                INT_DATA_TYPE)
        return new_tensor, mask

    def encode(self, inputs, pad_or_truncate=False):
        '''
        Input:
        inputs (list[str]) - list of strings
        pad_or_truncate (bool) - boolean for whether to perform padding
        (if sequence of tokens is shorter than self.max_length) or truncation
        (if sequence of tokens is longer than self.max_length)

        Output:
        list of tokenized strings (list[torch.Tensor[int]])
        list of masks (None or list[torch.Tensor[int]])
        '''
        # collect int tensors for each text input
        int_tensors = []
        for text in inputs:
            int_tensors.append(self._encode(text))
        masks = None

        # add integers corresponding to the padding token so
        # lengths of tensors all match, truncate if over max length
        if pad_or_truncate:
            masks = []
            for i in range(len(int_tensors)):
                int_tensor, mask = self.pad_or_truncate(int_tensors[i])
                int_tensors[i] = int_tensor.type(INT_DATA_TYPE)
                masks.append(mask.type(INT_DATA_TYPE))

        # return int tensors and masks
        return int_tensors, masks

    def _encode(self, text):
        '''
        Input:
        text (str) - a single string

        Output:
        tokenized string (torch.Tensor[int])
        '''
        # iterate through text, convert tokens to integers and return Tensor
        ints = []
        idx = 0
        while idx < len(text):
            # check for special token
            found_special_token = False
            for special_token in self.special_tokens:
                if text[idx:idx + len(special_token)] == special_token:
                    ints.append(self.token_to_int[special_token])
                    idx += len(special_token)
                    found_special_token = True
                    break
            if found_special_token:
                continue

            # check for word token
            # first, find the earliest word separator (could be special token
            # or non-alphanumeric character)
            first_word_separator = None
            first_word_separator_idx = None
            word_separators = list(set(self.special_tokens).union(
                self.non_alnum_chars))
            for word_separator in word_separators:
                word_separator_idx = text[idx:].find(word_separator)
                if word_separator_idx != -1:
                    if first_word_separator is None or\
                            first_word_separator_idx > word_separator_idx:
                        first_word_separator = word_separator
                        first_word_separator_idx = word_separator_idx
            # extract word token
            word_token = None
            if first_word_separator is None:
                word_token = text[idx:]
            elif first_word_separator_idx != 0:
                word_token = text[idx:idx + first_word_separator_idx]
            # add corresponding integer to sequence if word token is present in
            # vocabulary
            if word_token is not None and\
                    word_token in self.token_to_int.keys():
                ints.append(self.token_to_int[word_token])
                idx += len(word_token)
                continue

            # if not special token or word token, proceed with character-level
            # tokenization
            ints.append(self.token_to_int[text[idx]])
            idx += 1
        return torch.Tensor(ints).type(INT_DATA_TYPE)

    def decode(self, int_tensors):
        '''
        Input:
        int_tensors (list[torch.Tensor[int]]) - tokenized strings

        Output:
        a list of strings, represented by the provided tokens (list(str))
        '''
        # decode each integer tensor, collect and return strings
        strs = []
        for ints in int_tensors:
            strs.append(self._decode(ints.type(INT_DATA_TYPE)))
        return strs

    def _decode(self, ints):
        '''
        Input:
        ints (torch.Tensor[int]) - a sequence of tokens

        Output:
        the string represented by the sequence of provided tokens (str)
        '''
        # convert tensor of integers to tokens, append to string
        str = ''
        for int in ints.tolist():
            token = self.int_to_token[int]
            str += token
        return str


class Poetron:
    '''
    A system that uses a custom generative pretrained transformer (GPT)
    language model to generate/complete short poems
    '''
    def __init__(self):
        # specify special tokens
        self.poem_start_token = '<poem_start>'
        self.line_end_token = '<line_end>'
        self.poem_end_token = '<poem_end>'
        self.padding_token = '<padding>'
        self.special_tokens = [
            self.padding_token, self.poem_start_token, self.line_end_token,
            self.poem_end_token]

        # set context size
        self.context_size = 400

        # set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def _get_dataset(self):
        # download three line poem dataset, remove rows with missing values
        path = kagglehub.dataset_download('hjhalani30/haiku-dataset')
        dataset_df = pd.read_csv(os.path.join(path, 'all_haiku.csv'))
        dataset_df = dataset_df.dropna().reset_index(drop=True)

        # consolidate columns to get full poem, with special tokens
        dataset_df['poem'] = self.poem_start_token\
            + dataset_df['0'].str.strip().str.lower() + self.line_end_token\
            + dataset_df['1'].str.strip().str.lower() + self.line_end_token\
            + dataset_df['2'].str.strip().str.lower() + self.poem_end_token
        self.dataset_df = dataset_df[['poem']].reset_index(drop=True)

    def _get_tokenizer(self):
        # get character tokens from dataset (ignoring special tokens)
        chars = set()
        for i in range(len(self.dataset_df['poem'])):
            poem = self.dataset_df['poem'][i]
            for special_token in self.special_tokens:
                poem = poem.replace(special_token, '')
            chars = chars.union(set(poem))
        
        # split poems on non-alphanumeric tokens (ignoring special tokens)
        # to get word tokens
        words = set()
        non_alnum_chars = set(filter(lambda c: not c.isalnum(), chars))
        for i in range(len(self.dataset_df['poem'])):
            poem = self.dataset_df['poem'][i]
            for special_token in self.special_tokens:
                poem = poem.replace(special_token, ' ')
            for non_alnum_char in non_alnum_chars:
                poem = poem.replace(non_alnum_char, ' ')
            words = words.union(set(poem.split(' ')))

        # combine sets of word and character tokens
        tokens = sorted(chars.union(words))

        # map tokens to integers, and vice versa
        ints = list(range(len(tokens)))
        token_to_int = {token: int for token, int in zip(tokens, ints)}
        int_to_token = {int: token for int, token in zip(ints, tokens)}

        # map special tokens to integers
        for i in range(len(self.special_tokens)):
            special_token = self.special_tokens[i]
            token_to_int[special_token] = len(tokens) + i
            int_to_token[len(tokens) + i] = special_token

        # create tokenizer
        self.tokenizer = PoetronTokenizer(
            self.special_tokens, self.padding_token, token_to_int,
            int_to_token, self.context_size, non_alnum_chars)

    # utility function to get batch of training data for pretraining
    def _get_batch(self, batch_size):
        '''
        Input:
        batch_size (int) - number of (tokens, next token) pairs to generate
        and return

        Output:
        batch_inputs - torch.Tensor[int], with shape (batch_size,
        self.context_size)
        batch_input_masks - torch.Tensor[int], with shape (batch_size,
        self.context_size)
        batch_next_tokens - torch.Tensor[int], with shape (batch_size,
        self.context_size)
        '''
        # sample poems
        idxs = random.choices(
            range(len(self.dataset_df)), k=batch_size)
        poems = self.dataset_df.iloc[idxs]['poem'].to_list()
        tokenized_poems, _ = self.tokenizer.encode(poems)

        # create inputs and next token pairs
        batch_inputs = []
        batch_next_tokens = []
        batch_input_masks = []
        for tok_poem in tokenized_poems:
            # get input tokens and the corresponding next tokens
            input_tokens = tok_poem[:-1]
            next_tokens = tok_poem[1:]

            # pad/truncate input tokens and next tokens
            input_tokens, mask = self.tokenizer.pad_or_truncate(input_tokens)
            next_tokens, _ = self.tokenizer.pad_or_truncate(next_tokens)

            # add to batch
            batch_inputs.append(input_tokens)
            batch_input_masks.append(mask)
            batch_next_tokens.append(next_tokens)

        batch_inputs = torch.stack(batch_inputs, dim=0).type(INT_DATA_TYPE)
        batch_input_masks = torch.stack(batch_input_masks, dim=0).type(
            INT_DATA_TYPE)
        batch_next_tokens = torch.stack(batch_next_tokens, dim=0).type(
            INT_DATA_TYPE)

        # return batch of samples
        return batch_inputs, batch_input_masks, batch_next_tokens

    def _reshape_logits_and_next_toks(
            self, batch_logits, batch_input_masks, batch_next_tokens):
        '''
        Input:
        batch_logits (torch.Tensor[float]) - logits obtained from forward pass
        through model, shape is (batch size, context size, vocab size)
        batch_input_mask (torch.Tensor[int]) - input masks for each batch item,
        shape is (batch size, context size, vocab size)
        batch_next_tokens (torch.Tensor[int]) - next tokens of the batch,
        shape is (batch size, context size)

        Output:
        reshaped_logits (torch.Tensor[float]) - reshaped logits, shape is
        (at most batch size * context size, vocab size)
        reshaped_next_tokens (torch.Tensor[float]) - reshaped next tokens, shape
        is (at most batch size * context size)
        '''

        # reshape logits and next tokens to get more training samples
        reshaped_logits = []
        reshaped_next_tokens = []
        for i in range(batch_logits.shape[0]):
            input_mask = batch_input_masks[i]
            logits = batch_logits[i]
            next_tokens = batch_next_tokens[i]

            # if input mask is all zeros, only take the logits/next token
            # corresponding to the final token
            if torch.equal(input_mask, torch.zeros(
                    self.context_size, device=self.device)):
                reshaped_logits.append(logits[-1:])
                reshaped_next_tokens.append(next_tokens[-1:])
            # otherwise, take logits/next tokens for all tokens except the
            # padding tokens at the start of the sequence
            else:
                first_nonzero_idx = torch.argmax(input_mask)
                reshaped_logits.append(logits[first_nonzero_idx:])
                reshaped_next_tokens.append(
                    next_tokens[first_nonzero_idx:])
        reshaped_logits = torch.cat(reshaped_logits, dim=0)
        reshaped_next_tokens = torch.cat(reshaped_next_tokens, dim=0)

        # return reshaped logits and next tokens
        return reshaped_logits, reshaped_next_tokens

    def pretrain(self, batch_size, iters=1000, lr=1e-3, log_iters=100):
        '''
        Input:
        batch_size (int) - size of each batch of input token sequences and
        their corresponding next tokens
        epochs (int) - number of pretraining epochs
        '''
        # get dataset and create tokenizer
        self._get_dataset()
        self._get_tokenizer()

        # instantiate model
        self.model = PoetronModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=64,
            context_size=self.context_size,
            num_attn_heads=4,
            attn_head_size=32,
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_blocks=6,
            device=self.device
        ).to(self.device)
        self.model.train()

        # create optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)

        # pretrain model
        for iter in tqdm(range(iters)):
            # reset gradients
            optimizer.zero_grad()

            # get batch
            batch_inputs, batch_input_masks, batch_next_tokens = \
                self._get_batch(batch_size)
            batch_inputs = batch_inputs.to(self.device)
            batch_input_masks = batch_input_masks.to(self.device)
            batch_next_tokens = batch_next_tokens.to(self.device)

            # get logits (enriched token embeddings, projected to vocab_size
            # dimensions)
            batch_logits = self.model(batch_inputs, batch_input_masks)

            # reshape logits to get more training examples
            reshaped_logits, reshaped_next_tokens = \
                self._reshape_logits_and_next_toks(
                    batch_logits, batch_input_masks, batch_next_tokens)

            # calculate loss
            loss = F.cross_entropy(
                reshaped_logits, reshaped_next_tokens.type(torch.long),
                reduction='mean')

            # backpropagate loss to update weights
            loss.backward()
            optimizer.step()

            # after a certain number of iterations, print loss and generate
            # a short poem (use the generate method)
            if (iter + 1) % log_iters == 0:
                sample_poem = self.generate([''], self.context_size - 1,
                                            postprocess=True)[0]
                print(f'Average loss across batch in iteration {iter + 1}: '
                       +f'{loss.item()}.\nSample Poem:\n{sample_poem}')
        self.model.eval()

    @torch.no_grad()
    def generate(self, input_texts, max_new_tokens, postprocess=False):
        '''
        Input:
        input_texts (list[str]) - list of input poem texts to complete

        Output:
        output_texts (list[str]) - list of completed poem texts
        '''

        # if any input texts are empty, replace with poem start token
        input_texts = list(map(
            lambda t: t if len(t) != 0 else self.poem_start_token,
            input_texts))

        # tokenize inputs, get masks
        # shapes are (batch size, context size)
        tokenized_inputs, input_masks = self.tokenizer.encode(
            input_texts, pad_or_truncate=True)
        tokenized_inputs = torch.stack(tokenized_inputs, dim=0).to(self.device)
        input_masks = torch.stack(input_masks, dim=0).to(self.device)
        
        # repeatedly pass through model until max new tokens have been generated
        new_tokens_generated = 0
        while new_tokens_generated < max_new_tokens:
            # get logits for final tokens (shape is batch size, vocab size)
            logits = self.model(tokenized_inputs, input_masks)[:, -1]

            # use softmax function to get probability distribution over
            # vocabulary (shape is batch size, vocab size)
            next_token_dists = F.softmax(logits, dim=1)

            # sample from distributions to get next tokens (shape is batch size,
            # 1)
            next_token_ints = torch.multinomial(next_token_dists, num_samples=1)

            # add next token ints to input and update masks, shapes are
            # (batch size, context size + 1)
            tokenized_inputs = torch.cat([
                tokenized_inputs, next_token_ints], dim=1)
            input_masks = torch.cat([
                input_masks, torch.ones_like(
                    next_token_ints, device=self.device)], dim=1)

            # remove first tokens in the sequence and update masks, to bring
            # them back to shape (batch size, context size)
            tokenized_inputs = tokenized_inputs[:, 1:]
            input_masks = input_masks[:, 1:]

            # update new tokens generated
            new_tokens_generated += 1

        # decode tokens
        tokenized_inputs = tokenized_inputs.detach().cpu()
        int_tensors = []
        for i in range(tokenized_inputs.shape[0]):
            int_tensors.append(tokenized_inputs[i])
        generated_texts = self.tokenizer.decode(int_tensors)

        # postprocess generated texts
        if postprocess:
            generated_texts = list(map(
                lambda t:
                t[t.find(self.poem_start_token) : t.find(self.poem_end_token)]\
                    .replace(self.padding_token, '')\
                    .replace(self.poem_start_token, '')\
                    .replace(self.poem_end_token, '')\
                    .replace(self.line_end_token, '\n'), generated_texts))

        # return texts
        return generated_texts
