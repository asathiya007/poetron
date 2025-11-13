import kagglehub
import os
import pandas as pd
from poetron_model import PoetronModel
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


# set integer data type
INT_DATA_TYPE = torch.int32


class PoetronTokenizer:
    '''
    Tokenizer for the Poetron foundation language model.
    '''
    def __init__(self, special_tokens, padding_token, token_to_int,
                 int_to_token, max_length):
        self.special_tokens = special_tokens
        self.padding_token = padding_token
        self.token_to_int = token_to_int
        self.int_to_token = int_to_token
        self.vocab_size = len(self.token_to_int.keys())
        self.max_length = max_length

    def pad_or_truncate(self, tensor):
        '''
        Args:
        tensor (torch.Tensor) - a sequence of tokens to pad/truncate

        Returns:
        new_tensor (torch.Tensor[int]) - the sequence of tokens that has
        either been padded (if length < self.max_length) or truncated (if
        length > self.max_length), with shape (self.max_length)
        '''
        if len(tensor) < self.max_length:
            new_tensor = torch.cat(
                [torch.ones(self.max_length - len(tensor))
                    * self.token_to_int[self.padding_token], tensor])
        elif len(tensor) > self.max_length:
            new_tensor = tensor[-self.max_length:]
        return new_tensor.type(INT_DATA_TYPE)

    def encode(self, inputs, pad_or_truncate=False):
        '''
        Args:
        inputs (list[str]) - list of strings
        pad_or_truncate (bool) - boolean for whether to perform padding
        (if sequence of tokens is shorter than self.max_length) or truncation
        (if sequence of tokens is longer than self.max_length)

        Returns:
        list of tokenized strings (list[torch.Tensor[int]])
        '''
        # collect int tensors for each text input
        int_tensors = []
        for text in inputs:
            int_tensors.append(self._encode(text))

        # add integers corresponding to the padding token so
        # lengths of tensors all match, truncate if over max length
        if pad_or_truncate:
            for i in range(len(int_tensors)):
                int_tensors[i] = self.pad_or_truncate(int_tensors[i]).type(
                    INT_DATA_TYPE)

        # return tensors
        return int_tensors

    def _encode(self, text):
        '''
        Args:
        text (str) - a single string

        Returns:
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
            if found_special_token:
                continue

            # if not special token, proceed with character-level tokenization
            ints.append(self.token_to_int[text[idx]])
            idx += 1
        return torch.Tensor(ints).type(INT_DATA_TYPE)

    def decode(self, int_tensors):
        '''
        Args:
        int_tensors (list[torch.Tensor[int]]) - tokenized strings

        Returns:
        a list of strings, represented by the provided tokens (list(str))
        '''
        # decode each integer tensor, collect and return strings
        strs = []
        for ints in int_tensors:
            strs.append(self._decode(ints.type(INT_DATA_TYPE)))
        return strs

    def _decode(self, ints):
        '''
        Args:
        ints (torch.Tensor[int]) - a sequence of tokens

        Returns:
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
    A system that uses a language model to generate/complete three-line poems
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
        self.context_size = 240

    def _get_dataset(self):
        # download three line poem dataset, remove rows with missing values
        path = kagglehub.dataset_download('hjhalani30/haiku-dataset')
        dataset_df = pd.read_csv(os.path.join(path, 'all_haiku.csv'))
        dataset_df = dataset_df.dropna().reset_index(drop=True)

        # consolidate columns to get full poem, with special tokens
        dataset_df['poem'] = self.poem_start_token + dataset_df['0']\
            + self.line_end_token + dataset_df['1'] + self.line_end_token\
            + dataset_df['2'] + self.poem_end_token
        self.dataset_df = dataset_df[['poem']].reset_index(drop=True)

    def _get_tokenizer(self):
        # get characters in dataset (ignoring special tokens) for
        # character-based tokenization
        chars = set()
        for i in range(len(self.dataset_df['poem'])):
            poem = self.dataset_df['poem'][i]
            for special_token in self.special_tokens:
                poem = poem.replace(special_token, '')
            chars = chars.union(set(poem))
        chars = sorted(chars)

        # map character tokens to integers, and vice versa
        ints = list(range(len(chars)))
        token_to_int = {char: int for char, int in zip(chars, ints)}
        int_to_token = {int: char for int, char in zip(ints, chars)}

        # map special tokens to integers
        for i in range(len(self.special_tokens)):
            special_token = self.special_tokens[i]
            token_to_int[special_token] = len(chars) + i
            int_to_token[len(chars) + i] = special_token

        # create tokenizer
        self.tokenizer = PoetronTokenizer(
            self.special_tokens, self.padding_token, token_to_int,
            int_to_token, self.context_size)

    # utility function to get batch of training data for pretraining
    def _get_batch(self, batch_size):
        '''
        Args:
        batch_size (int) - number of (tokens, next token) pairs to generate
        and return

        Returns:
        batch_inputs - torch.Tensor[int], with shape (batch_size,
        self.context_size)
        batch_next_tokens - torch.Tensor[int], with shape (batch_size)
        '''
        # sample poems
        idxs = random.choices(
            range(len(self.dataset_df)), k=batch_size)
        poems = self.dataset_df.iloc[idxs]['poem'].to_list()
        tokenized_poems = self.tokenizer.encode(poems)

        # create inputs and next token pairs
        batch_inputs = []
        batch_next_tokens = []
        for tok_poem in tokenized_poems:
            # get slice of input tokens and the next token
            split_idx = random.choice(range(len(tok_poem) - 1))
            input_tokens = tok_poem[:split_idx + 1]
            next_token = tok_poem[split_idx + 1]

            # pad/truncate input tokens
            input_tokens = self.tokenizer.pad_or_truncate(input_tokens)

            # add to batch
            batch_inputs.append(input_tokens)
            batch_next_tokens.append(next_token)

        batch_inputs = torch.stack(batch_inputs, dim=0).type(INT_DATA_TYPE)
        batch_next_tokens = torch.Tensor(batch_next_tokens).type(INT_DATA_TYPE)

        # return batch of samples
        return batch_inputs, batch_next_tokens

    def pretrain(self, batch_size, epochs=1000):
        '''
        Args:
        batch_size (int) - size of each batch of input token sequences and
        their corresponding next tokens
        epochs (int) - number of pretraining epochs
        '''
        # get dataset and create tokenizer
        self._get_dataset()
        self._get_tokenizer()

        # instantiate model
        model = PoetronModel()
        model.train()

        # create optimizer
        optimizer = AdamW()

        # pretrain model
        for _ in tqdm(range(epochs)):
            # get batch
            batch_inputs, batch_next_tokens = self._get_batch(batch_size)

            # get predicted next token distributions
            pred_next_token_dists = model(batch_inputs, batch_next_tokens,
                                          self.tokenizer)

            # calculate loss
            loss = nn.CrossEntropyLoss(
                pred_next_token_dists, batch_next_tokens, reduction='mean')

            # backpropagate loss to update weights
            loss.backward()
            optimizer.step()
        model.eval()
        self.model = model
