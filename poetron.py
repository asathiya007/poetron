import kagglehub
import logging
import os
import pandas as pd
import pickle
from poetron_model import PoetronModel
from profanity_check import predict as predict_profanity
import random
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


# set integer data type
INT_DATA_TYPE = torch.int32

# set default save/load directory
DEFAULT_SAVE_DIR = './poetron_save_dir'


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
            new_tensor = tensor[-self.max_length:].clone()
        else:
            new_tensor = tensor.clone()
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
        self.context_size = 250

        # set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # get logger
        self.logger = logging.getLogger('Poetron_Logger')
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(stream=sys.stdout)

    def _get_dataset(self):
        self.logger.info('Downloading and processing datasets...')

        # download dataset, remove rows with missing values
        path = kagglehub.dataset_download('hjhalani30/haiku-dataset')
        dataset_df1 = pd.read_csv(os.path.join(path, 'all_haiku.csv'))
        dataset_df1 = dataset_df1.dropna().reset_index(drop=True)
        # consolidate columns to get full poem, with special tokens
        dataset_df1['poem'] = self.poem_start_token\
            + dataset_df1['0'].str.strip().str.lower() + self.line_end_token\
            + dataset_df1['1'].str.strip().str.lower() + self.line_end_token\
            + dataset_df1['2'].str.strip().str.lower() + self.poem_end_token
        # check for profanity
        poems = list(filter(
            lambda p: int(predict_profanity([p])[0]) == 0,
            list(dataset_df1['poem'])))
        dataset_df1 = pd.DataFrame({'poem': poems})

        # download dataset and process poems
        path = kagglehub.dataset_download('bfbarry/haiku-dataset')
        dataset2 = []
        with open(os.path.join(path, 'lines.txt'), 'r') as poems_file:
            for poem in poems_file:
                # check for profanity
                contains_profanity = int(predict_profanity([poem])[0])
                if contains_profanity == 1:
                    continue

                # process poem and check for empty lines
                proc_poem = poem.lower().replace('$', '')
                proc_poem_lines = proc_poem.split('/')
                empty_line_present = False
                for i in range(len(proc_poem_lines)):
                    proc_poem_lines[i] = proc_poem_lines[i].strip()
                    if len(proc_poem_lines[i]) == 0:
                        empty_line_present = True
                        break
                if empty_line_present:
                    continue
                proc_poem = self.poem_start_token\
                    + self.line_end_token.join(proc_poem_lines)\
                    + self.poem_end_token
                dataset2.append(proc_poem)
        dataset_df2 = pd.DataFrame({'poem': dataset2})

        # combine datasets
        self.dataset_df = pd.concat(
            [dataset_df1, dataset_df2], axis=0, ignore_index=True)
        self.logger.info('Downloaded and processed datasets')

    def _get_tokenizer(self):
        self.logger.info('Creating tokenizer...')

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
        self.logger.info('Created tokenizer')

    # utility function to get batch of training data for pretraining
    def _get_batch(self, batch_size, start_idx=None):
        '''
        Input:
        batch_size (int) - number of (tokens, next token) pairs to generate
        and return
        start_idx (int or None) - the starting index of the batch (if None,
        random indices are used)

        Output:
        batch_inputs (torch.Tensor[int]) - tensor of input tokens, shape is
        (batch size, context size)
        batch_input_masks (torch.Tensor[int]) - tensor of input masks, shape
        is (batch size, context size)
        batch_next_tokens (torch.Tensor[int]) - tensor of next tokens for
        every contiguous subsequence of tokens in batch_inputs, shape is
        (batch size, context size)
        '''
        # sample poems
        if start_idx is not None:
            idxs = list(range(
                start_idx, min(start_idx + batch_size, len(self.dataset_df))))
        else:
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
    
    def _get_model_instance(self):
        '''
        Input:
        None

        Output:
        model (PoetronModel) - an instance of the PoetronModel class
        '''
        self.logger.info('Instantiating model...')

        # create an instance of the PoetronModel class
        model = PoetronModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=64,
            context_size=250,
            num_attn_heads=4,
            attn_head_size=32,
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_blocks=6,
            device=self.device).to(self.device)
        self.logger.info('Instantiated model')

        # print number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'Model has {num_params / 1e6}M parameters')

        # return PoetronModel instance
        return model


    def pretrain(self, batch_size, epochs=5, log_epochs=1):
        '''
        Input:
        batch_size (int) - size of each batch of input token sequences and
        their corresponding next tokens
        epochs (int) - number of pretraining epochs
        log_epochs (int) - number of epochs after which to print loss and
        generate a sample poem
        '''
        # get dataset and create tokenizer
        self._get_dataset()
        self._get_tokenizer()

        # instantiate model
        self.model = self._get_model_instance()
        self.model.train()

        # create optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-4)

        def _pretrain_fw_pass(
                batch_inputs, batch_input_masks, batch_next_tokens):
            '''
            Input:
            batch_inputs (torch.Tensor[int]) - tensor of input tokens, shape is
            (batch size, context size)
            batch_input_masks (torch.Tensor[int]) - tensor of input masks, shape
            is (batch size, context size)
            batch_next_tokens (torch.Tensor[int]) - tensor of next tokens for
            every contiguous subsequence of tokens in batch_inputs, shape is
            (batch size, context size)

            Output:
            loss (torch.Tensor[float]) - loss value obtained from output of
            forward pass with the given batch of inputs, input masks, and next
            tokens
            '''
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
            
            # return loss
            return loss
        
        @torch.no_grad()
        def _get_eval_loss():
            '''
            Input:
            None

            Output:
            loss (torch.Tensor[float]) - returns loss of a pretraining forward
            pass on a random batch of samples
            '''

            # get batch of random samples
            batch_inputs, batch_input_masks, batch_next_tokens = \
                self._get_batch(batch_size, None)
            batch_inputs = batch_inputs.to(self.device)
            batch_input_masks = batch_input_masks.to(self.device)
            batch_next_tokens = batch_next_tokens.to(self.device)
            
            # get and return loss
            loss = _pretrain_fw_pass(
                batch_inputs, batch_input_masks, batch_next_tokens)
            return loss

        # pretrain model
        self.logger.info('Pretraining model...')
        for epoch in range(epochs):
            batch_start_idxs = list(range(0, len(self.dataset_df), batch_size))
            for batch_start_idx in tqdm(
                batch_start_idxs, desc=f'Epoch {epoch + 1} of {epochs}'):
                # reset gradients
                optimizer.zero_grad()

                # get batch
                batch_inputs, batch_input_masks, batch_next_tokens = \
                    self._get_batch(batch_size, batch_start_idx)
                batch_inputs = batch_inputs.to(self.device)
                batch_input_masks = batch_input_masks.to(self.device)
                batch_next_tokens = batch_next_tokens.to(self.device)

                # perform a pretraining forward pass, get loss
                loss = _pretrain_fw_pass(
                    batch_inputs, batch_input_masks, batch_next_tokens)

                # backpropagate loss to update weights
                loss.backward()
                optimizer.step()

                # update batch start index
                batch_start_idx += batch_size

            # after a certain number of iterations, print loss and generate
            # a short poem (use the generate method)
            if (epoch + 1) % log_epochs == 0:
                sample_poem = self.generate([''], self.context_size - 1,
                                            postprocess=True)[0]
                eval_loss = _get_eval_loss()
                self.logger.info(
                    'Average loss across batch of random samples after epoch '
                    + f'{epoch + 1}: {eval_loss.item()}.\nSample Poem:\n'
                    + f'{sample_poem}')
        self.model.eval()
        self.logger.info('Finished pretraining model')

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

    def save(self, save_dir=DEFAULT_SAVE_DIR):
        # create save directory if it doesn't already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # save tokenizer
        tokenizer_save_path = os.path.join(save_dir, 'tokenizer.pkl')
        with open(tokenizer_save_path, 'wb') as tokenizer_file:
            pickle.dump(self.tokenizer, tokenizer_file)
        self.logger.info(f'Saved tokenizer to {tokenizer_save_path}')

        # save model
        model_save_path = os.path.join(save_dir, 'model_state_dict.pth')
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(
            f'Saved model parameters and buffers to {model_save_path}')

    def load(self, load_dir=DEFAULT_SAVE_DIR):
        # load tokenizer
        tokenizer_save_path = os.path.join(load_dir, 'tokenizer.pkl')
        with open(tokenizer_save_path, 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)
        self.logger.info(f'Loaded tokenizer from {tokenizer_save_path}')

        # load model
        model_save_path = os.path.join(load_dir, 'model_state_dict.pth')
        self.model = self._get_model_instance()
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.eval()
        self.logger.info(
            f'Loaded model parameters and buffers from {model_save_path}')
