from poetron import Poetron, INT_DATA_TYPE
from poetron_model import SelfAttnHead, MultiHeadAttn, FeedFwd, AttnBlock, \
    PoetronModel
import torch


# placeholder constants
PLACEHOLDER_EMBED_DIM = 20
PLACEHOLDER_ATTN_HEAD_SIZE = 10
PLACEHOLDER_NUM_ATTN_HEADS = 5
PLACEHOLDER_HIDDEN_SIZE = 256
PLACEHOLDER_NUM_HIDDEN_LAYERS = 2
PLACEHOLDER_NUM_ATTN_BLOCKS = 8
PLACEHOLDER_VOCAB_SIZE = 300


def test_get_dataset():
    # get dataset DataFrame
    p = Poetron()
    p._get_dataset()
    dataset_df = p.dataset_df

    # check dataset columns
    dataset_cols = dataset_df.columns.to_list()
    exp_cols = ['poem']
    assert dataset_cols == exp_cols, 'Incorrect dataset columns. Expected: '\
        + f'{exp_cols}. Actual: {dataset_cols}'

    # check each row of the dataset
    poem_start_token = p.poem_start_token
    line_end_token = p.line_end_token
    poem_end_token = p.poem_end_token
    for _, row in dataset_df.iterrows():
        poem = row['poem']

        # check that the row starts with the poem start token
        assert poem[:len(poem_start_token)] == poem_start_token, \
            'Poem does not start with the poem start token '\
            + f'{poem_start_token}. Poem: {poem}'

        # check that the row ends with the poem end token
        assert poem[-len(poem_end_token):] == poem_end_token, \
            'Poem does not end with the poem end token '\
            + f'{poem_end_token}. Poem: {poem}'

        # check that there are three non-empty lines of the poem, separated by
        # the line end token
        assert line_end_token in poem, 'Poem does not contain line end '\
            + f'tokens {line_end_token}. Poem: {poem}'
        poem_lines = poem.replace(poem_start_token, '').replace(
            poem_end_token, '').split(line_end_token)
        exp_num_lines = 3
        act_num_lines = len(poem_lines)
        assert exp_num_lines == act_num_lines, 'Poem contains incorrect '\
            + f'number of lines. Expected: {exp_num_lines}. Actual: '\
            + f'{act_num_lines}. Poem: {poem}'
        for line in poem_lines:
            assert len(line) > 0, 'Poem contains an empty line(s). Poem: '\
                + f'{poem}'


def test_get_tokenizer():
    # get tokenizer
    p = Poetron()
    p._get_dataset()
    p._get_tokenizer()
    tokenizer = p.tokenizer

    # check max length
    assert p.context_size == tokenizer.max_length, 'Incorrect tokenizer max '\
        + f'length. Expected: {p.context_size}. Actual: {tokenizer.max_length}'

    # verify mappings
    token_to_int = tokenizer.token_to_int
    int_to_token = tokenizer.int_to_token
    # check data types of token to int mappings
    key_types = list(set(map(lambda k: type(k), token_to_int.keys())))
    value_types = list(set(map(lambda k: type(k), token_to_int.values())))
    assert len(key_types) == 1, 'Keys of token to int mappings are of '\
        + f'multiple types: {key_types}'
    assert len(value_types) == 1, 'Values of token to int mappings are of '\
        + f'multiple types: {value_types}'
    assert key_types[0] is str, 'Incorrect key type for token to '\
        + f'int mapping. Expected: str. Actual: {key_types[0]}'
    assert value_types[0] is int, 'Incorrect value type for token to '\
        + f'int mapping. Expected: int. Actual: {value_types[0]}'
    # check data types of int to token mappings
    key_types = list(set(map(lambda k: type(k), int_to_token.keys())))
    value_types = list(set(map(lambda k: type(k), int_to_token.values())))
    assert len(key_types) == 1, 'Keys of token to int mappings are of '\
        + f'multiple types: {key_types}'
    assert len(value_types) == 1, 'Values of token to int mappings are of '\
        + f'multiple types: {value_types}'
    assert key_types[0] is int, 'Incorrect key type for int to '\
        + f'token mapping. Expected: int. Actual: {key_types[0]}'
    assert value_types[0] is str, 'Incorrect value type for int to '\
        + f'token mapping. Expected: str. Actual: {value_types[0]}'
    # check that mappings are the same length and one-to-one
    assert len(token_to_int) == len(int_to_token), 'Number of token to int '\
        + f'mappings ({len(token_to_int)}) is not equal to number of int to '\
        + f'token mappings ({len(int_to_token)})'
    num_int_values = len(token_to_int.values())
    assert num_int_values == len(token_to_int), 'Token to int mapping is not '\
        + 'one-to-one'
    num_token_values = len(int_to_token.values())
    assert num_token_values == len(int_to_token), 'Int to token mapping is '\
        + 'not one-to-one'
    # check that int to token mapping is reversed token to int mapping
    reversed_int_to_token = {token: int for int, token in int_to_token.items()}
    assert reversed_int_to_token == token_to_int, 'Reversed int to token '\
        + 'mapping is not equal to token to int mapping'

    # confirm that original text is recovered after encoding and then decoding
    # check number of tokens as well
    texts = [
        # tuple(text, expected number of tokens)
        ('The quick brown fox jumped over the lazy dog.', 45),
        ('What a great song that was! What is it called?', 46),
        ('<padding><padding><poem_start>Testing<line_end>Testing<line_end>'
         + '<poem_end>', 20)
    ]
    for text, exp_num_tokens in texts:
        int_tensors, masks = tokenizer.encode([text])

        # check tensor data types
        for int_tensor in int_tensors:
            assert int_tensor.dtype is INT_DATA_TYPE, 'Incorrect data type '\
                + f'for tokens tensor. Expected: {INT_DATA_TYPE}. Actual: '\
                + f'{int_tensor.dtype}'

        # check shape of encoding and correctness of decoded text
        exp_shape = (1, exp_num_tokens)
        assert torch.stack(int_tensors, dim=0).shape == exp_shape, \
            f'Incorrect shape of encoding output. Expected: {exp_shape}. '\
            + f'Actual: {int_tensors.shape}'
        decoded_text = tokenizer.decode(int_tensors)
        exp_texts = [text]
        assert decoded_text == exp_texts, 'Incorrect decoding output. '\
            + f'Expected: {exp_texts}. Actual: {decoded_text}'
        
        # check mask data types
        for mask in masks:
            assert mask.dtype is INT_DATA_TYPE, 'Incorrect data type '\
                + f'for mask tensor. Expected: {INT_DATA_TYPE}. Actual: '\
                + f'{mask.dtype}'

    # check that batch of encoded inputs are padded to the max length
    int_tensors, masks = tokenizer.encode(
        list(map(lambda t: t[0], texts)), pad_or_truncate=True)
    # check tensor data types
    for int_tensor in int_tensors:
        assert int_tensor.dtype is INT_DATA_TYPE, 'Incorrect data type '\
            + f'for tokens tensor. Expected: {INT_DATA_TYPE}. Actual: '\
            + f'{int_tensor.dtype}'
    # check mask data types
    for mask in masks:
        assert mask.dtype is INT_DATA_TYPE, 'Incorrect data type '\
            + f'for mask tensor. Expected: {INT_DATA_TYPE}. Actual: '\
            + f'{mask.dtype}'
    # check shape of encoded inputs, shape of masks, number of padding tokens,
    # and correctness of mask
    int_tensors = torch.stack(int_tensors, dim=0)
    masks = torch.stack(masks, dim=0)
    exp_shape = (len(texts), tokenizer.max_length)
    assert int_tensors.shape == exp_shape, 'Incorrect shape of encoding '\
        + f'output. Expected: {exp_shape}. Actual: {int_tensors.shape}'
    assert masks.shape == exp_shape, 'Incorrect shape of encoding '\
        + f'output. Expected: {exp_shape}. Actual: {masks.shape}'
    for i in range(len(int_tensors)):
        int_tensor = int_tensors[i]
        padding_tokens = set(
            int_tensor[:tokenizer.max_length - texts[i][1]].tolist())
        exp_padding_tokens = {tokenizer.token_to_int[tokenizer.padding_token]}
        assert padding_tokens == exp_padding_tokens, 'Incorrect padding '\
            f'tokens. Expected: {exp_padding_tokens}. Actual: {padding_tokens}'
        mask = masks[i]
        exp_mask = (int_tensor != tokenizer.token_to_int[
            tokenizer.padding_token]).type(torch.int32)
        assert torch.equal(exp_mask, mask), f'Incorrect mask. Expected: '\
            + f'{exp_mask}. Actual: {mask}'


def test_get_batch():
    # create Poetron instance, get dataset and tokenizer
    p = Poetron()
    p._get_dataset()
    p._get_tokenizer()

    # check batch inputs and next tokens have expected shape and data type
    batch_sizes = [2, 4, 8, 16, 32, 64]
    for batch_size in batch_sizes:
        batch_inputs, batch_input_masks, batch_next_tokens = p._get_batch(
            batch_size)

        # check shape of batch of inputs
        exp_shape = (batch_size, p.context_size)
        act_shape = batch_inputs.shape
        assert exp_shape == act_shape, 'Incorrect shape of batch of input '\
            + f'tokens. Expected: {exp_shape}. Actual: {act_shape}'

        # check that all token integers are present in the tokenizer's mapping
        token_ints = set(batch_inputs.flatten().tolist())
        mapping_ints = set(p.tokenizer.int_to_token.keys())
        other_ints = token_ints.difference(mapping_ints)
        assert len(other_ints) == 0, 'Batch of input tokens contains '\
            + 'integers outside of tokenizer\'s int to token mapping: '\
            + f'{other_ints}'

        # check shape of batch of input masks
        exp_shape = (batch_size, p.context_size)
        act_shape = batch_input_masks.shape
        assert exp_shape == act_shape, 'Incorrect shape of batch of input '\
            + f'masks. Expected: {exp_shape}. Actual: {act_shape}'
        
        # check that all values in the input mask are either 0 or 1
        mask_ints = set(batch_input_masks.flatten().tolist())
        exp_ints = {0, 1}
        other_ints = mask_ints.difference(exp_ints)
        assert len(other_ints) == 0, 'Batch of input masks contains '\
            + f'integers outside of {exp_ints}: {other_ints}'
        
        # check that mask is applied to all padding tokens
        exp_batch_mask = (batch_inputs != p.tokenizer.token_to_int[
            p.tokenizer.padding_token]).type(torch.int32)
        assert torch.equal(exp_batch_mask, batch_input_masks), 'Incorrect '\
            + f'batch of masks. Expected: {exp_batch_mask}. Actual: '\
            + f'{batch_input_masks}'

        # check shape of batch of next tokens
        exp_shape = (batch_size,)
        act_shape = batch_next_tokens.shape
        assert exp_shape == act_shape, 'Incorrect shape of batch of next '\
            + f'tokens. Expected: {exp_shape}. Actual: {act_shape}'

        # check that all token integers are present in the tokenizer's mapping
        token_ints = set(batch_next_tokens.flatten().tolist())
        other_ints = mapping_ints.difference(token_ints)
        assert len(other_ints) != 0, 'Batch of next tokens contains '\
            + 'integers outside of tokenizer\'s int to token mapping'


# helper function for checking attention patterns
def _check_attn_pattern(test_case_num, exp_attn_pattern, act_attn_pattern):
    assert torch.equal(exp_attn_pattern, act_attn_pattern),\
        'Expected and actual attention patterns don\'t match for test '\
        + f'case {test_case_num}. Expected: {exp_attn_pattern}, Actual: '\
        + f'{act_attn_pattern}'


def test_get_init_attn_pattern():
    TEST_CASES = [
        # tuple(attention head size, context size, tensor of query vectors,
        # tensor of key vectors, expected attention pattern)
        (
            torch.ones((4, 5, 10)),
            torch.ones((4, 5, 10)) * 2,
            torch.ones((4, 5, 5)) * 20
        ),
        (
            torch.ones((10, 4, 8)),
            torch.ones((10, 4, 8)) * 1.1,
            torch.ones((10, 4, 4)) * 8.8
        ),
        (
            torch.ones((1, 10, 4)) * 1.25,
            torch.ones((1, 10, 4)) * 2,
            torch.ones((1, 10, 10)) * 10
        ),
        (
            torch.cat([
                torch.ones((2, 5, 10)),
                torch.ones((2, 5, 10)) * 1.25
            ], dim=0),
            torch.cat([
                torch.ones((2, 5, 10)) * 2,
                torch.ones((2, 5, 10)) * 4,
            ], dim=0),
            torch.cat([
                torch.ones((2, 5, 5)) * 20,
                torch.ones((2, 5, 5)) * 50
            ], dim=0)
        )
    ]
    for i in range(len(TEST_CASES)):
        test_case = TEST_CASES[i]
        q, k, exp_attn_pattern = test_case
        attn_head_size = q.shape[2]
        context_size = q.shape[1]
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, attn_head_size, context_size)
        act_attn_pattern = sah._get_init_attn_pattern(q, k)
        _check_attn_pattern(i + 1, exp_attn_pattern, act_attn_pattern)


def test_scale_attn_pattern():
    TEST_CASES = [
        # tuple(attention pattern, scaling factor, expected scaled attention
        # pattern)
        (
            torch.ones((4, 5, 5)) * 20,
            0.6,
            torch.ones((4, 5, 5)) * 12,
        ),
        (
            torch.ones((10, 4, 4)) * 8.8,
            10,
            torch.ones((10, 4, 4)) * 88
        ),
        (
            torch.ones((1, 10, 10)) * 10,
            0.4,
            torch.ones((1, 10, 10)) * 4
        ),
        (
            torch.cat([
                torch.ones((2, 5, 5)) * 20,
                torch.ones((2, 5, 5)) * 50
            ], dim=0),
            0.1,
            torch.cat([
                torch.ones((2, 5, 5)) * 2,
                torch.ones((2, 5, 5)) * 5
            ], dim=0)
        )
    ]
    for i in range(len(TEST_CASES)):
        test_case = TEST_CASES[i]
        attn_pattern, scaling_factor, exp_scaled_attn_pattern = test_case
        context_size = attn_pattern.shape[1]
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size)
        act_scaled_attn_pattern = sah._scale_attn_pattern(
            attn_pattern, scaling_factor)
        _check_attn_pattern(
            i + 1, exp_scaled_attn_pattern, act_scaled_attn_pattern)


def test_apply_subseq_mask():
    TEST_CASES = [
        # tuple(attention pattern, expected attention pattern after masking
        # subsequent tokens)
        (
            torch.ones((4, 5, 5)) * 12,
            torch.stack([torch.Tensor([
                [12] * i + [float('-inf')] * (5 - i) for i in range(1, 6)
            ])] * 4, dim=0)
        ),
        (
            torch.ones((10, 4, 4)) * 88,
            torch.stack([torch.Tensor([
                [88] * i + [float('-inf')] * (4 - i) for i in range(1, 5)
            ])] * 10, dim=0)
        ),
        (
            torch.ones((1, 10, 10)) * 4,
            torch.stack([torch.Tensor([
                [4] * i + [float('-inf')] * (10 - i) for i in range(1, 11)
            ])], dim=0)
        ),
        (
            torch.cat([
                torch.ones((2, 5, 5)) * 2,
                torch.ones((2, 5, 5)) * 5
            ], dim=0),
            torch.cat([
                torch.stack([torch.Tensor([
                    [2] * i + [float('-inf')] * (5 - i) for i in range(1, 6)
                ])] * 2, dim=0),
                torch.stack([torch.Tensor([
                    [5] * i + [float('-inf')] * (5 - i) for i in range(1, 6)
                ])] * 2, dim=0)
            ], dim=0)
        )
    ]
    for i in range(len(TEST_CASES)):
        test_case = TEST_CASES[i]
        attn_pattern, exp_masked_attn_pattern = test_case
        context_size = attn_pattern.shape[1]
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size)
        act_masked_attn_pattern = sah._apply_subseq_mask(attn_pattern)
        _check_attn_pattern(
            i + 1, exp_masked_attn_pattern, act_masked_attn_pattern)


def test_apply_input_mask():
    TEST_CASES = [
        # tuple(attention pattern, input mask, expected attention pattern after
        # applying input mask)
        (
            torch.stack([torch.Tensor([
                [12] * i + [float('-inf')] * (5 - i) for i in range(1, 6)
            ])] * 4, dim=0),
            torch.stack([torch.Tensor([0] * 5)] * 4, dim=0),
            torch.stack([torch.Tensor([[float('-inf')] * 5] * 5)] * 4, dim=0)
        ),
        (
            torch.stack([torch.Tensor([
                [88] * i + [float('-inf')] * (4 - i) for i in range(1, 5)
            ])] * 10, dim=0),
            torch.stack([torch.Tensor([0, 1, 1, 1])] * 10, dim=0),
            torch.stack([torch.Tensor(
                [[float('-inf')] * 4]
                + [[float('-inf')] + [88] * i + [float('-inf')] * (3 - i)
                    for i in range(1, 4)]
            )] * 10, dim=0)
        ),
        (
            torch.stack([torch.Tensor([
                [4] * i + [float('-inf')] * (10 - i) for i in range(1, 11)
            ])], dim=0),
            torch.stack([torch.Tensor([1] * 10)], dim=0),
            torch.stack([torch.Tensor([
                [4] * i + [float('-inf')] * (10 - i) for i in range(1, 11)
            ])], dim=0),
        ),
        (
            torch.cat([
                torch.stack([torch.Tensor([
                    [2] * i + [float('-inf')] * (5 - i) for i in range(1, 6)
                ])] * 2, dim=0),
                torch.stack([torch.Tensor([
                    [5] * i + [float('-inf')] * (5 - i) for i in range(1, 6)
                ])] * 2, dim=0)
            ], dim=0),
            torch.stack(
                [torch.Tensor([0, 0, 1, 1, 1])] * 2
                + [torch.Tensor([0, 0, 0, 1, 1])] * 2, dim=0),
            torch.cat([
                torch.stack([torch.Tensor(
                    [[float('-inf')] * 5] * 2
                    + [[float('-inf')] * 2 + [2] * i + [float('-inf')] *
                        (3 - i) for i in range(1, 4)]
                )] * 2, dim=0),
                torch.stack([torch.Tensor(
                    [[float('-inf')] * 5] * 3
                    + [[float('-inf')] * 3 + [5] * i + [float('-inf')] *
                        (2 - i) for i in range(1, 3)]
                )] * 2, dim=0)
            ], dim=0)
        )
    ]
    for i in range(len(TEST_CASES)):
        test_case = TEST_CASES[i]
        attn_pattern, input_mask, exp_masked_attn_pattern = test_case
        context_size = attn_pattern.shape[1]
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size)
        act_masked_attn_pattern = sah._apply_input_mask(
            attn_pattern, input_mask)
        _check_attn_pattern(
            i + 1, exp_masked_attn_pattern, act_masked_attn_pattern)


def test_resolve_neg_inf_rows():
    TEST_CASES = [
        # tuple(attention pattern, expected attention pattern after resolving
        # rows with all -inf values)
        (
            torch.stack([torch.Tensor([[float('-inf')] * 5] * 5)] * 4, dim=0),
            torch.stack([torch.Tensor(
                [[float('-inf')] * i + [1] + [float('-inf')] * (4 - i)
                 for i in range(0, 5)]
            )] * 4, dim=0)
        ),
        (
            torch.stack([torch.Tensor(
                [[float('-inf')] * 4]
                + [[float('-inf')] + [88] * i + [float('-inf')] * (3 - i)
                    for i in range(1, 4)]
            )] * 10, dim=0),
            torch.stack([torch.Tensor(
                [[1] + [float('-inf')] * 3]
                + [[float('-inf')] + [88] * i + [float('-inf')] * (3 - i)
                    for i in range(1, 4)]
            )] * 10, dim=0)
        ),
        (
            torch.stack([torch.Tensor([
                [4] * i + [float('-inf')] * (10 - i) for i in range(1, 11)
            ])], dim=0),
            torch.stack([torch.Tensor([
                [4] * i + [float('-inf')] * (10 - i) for i in range(1, 11)
            ])], dim=0)
        ),
        (
            torch.cat([
                torch.stack([torch.Tensor(
                    [[float('-inf')] * 5] * 2
                    + [[float('-inf')] * 2 + [2] * i + [float('-inf')] *
                        (3 - i) for i in range(1, 4)]
                )] * 2, dim=0),
                torch.stack([torch.Tensor(
                    [[float('-inf')] * 5] * 3
                    + [[float('-inf')] * 3 + [5] * i + [float('-inf')] *
                        (2 - i) for i in range(1, 3)]
                )] * 2, dim=0)
            ], dim=0),
            torch.cat([
                torch.stack([torch.Tensor(
                    [[float('-inf')] * i + [1] + [float('-inf')] * (4 - i)
                     for i in range(0, 2)]
                    + [[float('-inf')] * 2 + [2] * i + [float('-inf')] *
                        (3 - i) for i in range(1, 4)]
                )] * 2, dim=0),
                torch.stack([torch.Tensor(
                    [[float('-inf')] * i + [1] + [float('-inf')] * (4 - i)
                     for i in range(0, 3)]
                    + [[float('-inf')] * 3 + [5] * i + [float('-inf')] *
                        (2 - i) for i in range(1, 3)]
                )] * 2, dim=0)
            ], dim=0)
        )
    ]
    for i in range(len(TEST_CASES)):
        test_case = TEST_CASES[i]
        attn_pattern, exp_resolved_attn_pattern = test_case
        context_size = attn_pattern.shape[1]
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size)
        act_resolved_attn_pattern = sah._resolve_neg_inf_rows(attn_pattern)
        _check_attn_pattern(
            i + 1, exp_resolved_attn_pattern, act_resolved_attn_pattern)


def test_normalize_attn_pattern():
    TEST_CASES = [
        # tuple(attention pattern, expected normalized attention pattern)
        (
            torch.stack([torch.Tensor(
                [[float('-inf')] * i + [1] + [float('-inf')] * (4 - i)
                    for i in range(0, 5)]
            )] * 4, dim=0),
            torch.stack([torch.eye(5)] * 4)
        ),
        (
            torch.stack([torch.Tensor(
                [[1] + [float('-inf')] * 3]
                + [[float('-inf')] + [88] * i + [float('-inf')] * (3 - i)
                    for i in range(1, 4)]
            )] * 10, dim=0),
            torch.stack([torch.Tensor(
                [[1] + [0] * 3] +
                [[0] + [1 / i] * i + [0] * (3 - i) for i in range(1, 4)])
            ] * 10, dim=0)
        ),
        (
            torch.stack([torch.Tensor([
                [4] * i + [float('-inf')] * (10 - i) for i in range(1, 11)
            ])], dim=0),
            torch.stack([torch.Tensor([
                [1 / i] * i + [0] * (10 - i) for i in range(1, 11)])
            ], dim=0)
        ),
        (
            torch.cat([
                torch.stack([torch.Tensor(
                    [[float('-inf')] * i + [1] + [float('-inf')] * (4 - i)
                     for i in range(0, 2)]
                    + [[float('-inf')] * 2 + [2] * i + [float('-inf')] *
                        (3 - i) for i in range(1, 4)]
                )] * 2, dim=0),
                torch.stack([torch.Tensor(
                    [[float('-inf')] * i + [1] + [float('-inf')] * (4 - i)
                     for i in range(0, 3)]
                    + [[float('-inf')] * 3 + [5] * i + [float('-inf')] *
                        (2 - i) for i in range(1, 3)]
                )] * 2, dim=0)
            ], dim=0),
            torch.cat([
                torch.stack([torch.Tensor(
                    [[0] * i + [1] + [0] * (4 - i) for i in range(0, 2)]
                    + [[0] * 2 + [1 / i] * i + [0] * (3 - i)
                       for i in range(1, 4)]
                )] * 2, dim=0),
                torch.stack([torch.Tensor(
                    [[0] * i + [1] + [0] * (4 - i) for i in range(0, 3)]
                    + [[0] * 3 + [1 / i] * i + [0] * (2 - i)
                       for i in range(1, 3)]
                )] * 2, dim=0)
            ], dim=0)
        )
    ]
    for i in range(len(TEST_CASES)):
        test_case = TEST_CASES[i]
        attn_pattern, exp_normalized_attn_pattern = test_case
        context_size = attn_pattern.shape[1]
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size)
        act_normalized_attn_pattern = sah._normalize_attn_pattern(attn_pattern)
        _check_attn_pattern(
            i + 1, exp_normalized_attn_pattern, act_normalized_attn_pattern)


# helper function for checking shape
def _check_shape(exp_shape, act_shape):
    assert exp_shape == act_shape, f'Incorrect shape. Expected: {exp_shape}. '\
        + f'Actual: {act_shape}.'


def test_self_attn_output_shape():
    TEST_CASES = [
        # tuple(embedding dimension, attention head size, context size,
        # batch size)
        (20, 16, 100, 4),
        (48, 24, 50, 10),
        (16, 32, 80, 10),
        (32, 48, 96, 4)
    ]
    for test_case in TEST_CASES:
        embed_dim, attn_head_size, context_size, batch_size = test_case
        input_mask = torch.stack([torch.ones(context_size)] * batch_size, dim=0)
        sah = SelfAttnHead(embed_dim, attn_head_size, context_size)
        sah_input = torch.randn((batch_size, context_size, embed_dim))
        sah_output = sah(sah_input, input_mask)
        exp_output_shape = (batch_size, context_size, attn_head_size)
        act_output_shape = sah_output.shape
        _check_shape(exp_output_shape, act_output_shape)


def test_get_concat_sah_outputs():
    TEST_CASES = [
        # tuple(embedding dimension, attention head size, context size,
        # batch size, number of attention heads)
        (20, 16, 100, 4, 12),
        (48, 24, 50, 10, 5),
        (16, 32, 80, 10, 1),
        (32, 48, 96, 4, 6)
    ]
    for test_case in TEST_CASES:
        embed_dim, attn_head_size, context_size, batch_size, num_attn_heads = \
            test_case
        input_mask = torch.stack([torch.ones(context_size)] * batch_size, dim=0)
        mha = MultiHeadAttn(
            embed_dim, context_size, num_attn_heads, attn_head_size)
        mha_input = torch.randn((batch_size, context_size, embed_dim))
        concat_sah_outputs = mha._get_concat_sah_outputs(mha_input, input_mask)
        exp_output_shape = (
            batch_size, context_size, attn_head_size * num_attn_heads)
        act_output_shape = concat_sah_outputs.shape
        _check_shape(exp_output_shape, act_output_shape)


def test_multi_head_attn_output_shape():
    TEST_CASES = [
        # tuple(embedding dimension, attention head size, context size,
        # batch size, number of attention heads)
        (20, 16, 100, 4, 12),
        (48, 24, 50, 10, 5),
        (16, 32, 80, 10, 1),
        (32, 48, 96, 4, 6)
    ]
    for test_case in TEST_CASES:
        embed_dim, attn_head_size, context_size, batch_size, num_attn_heads = \
            test_case
        input_mask = torch.ones((batch_size, context_size))
        mha = MultiHeadAttn(
            embed_dim, context_size, num_attn_heads, attn_head_size)
        mha_input = torch.randn((batch_size, context_size, embed_dim))
        mha_output = mha(mha_input, input_mask)
        exp_output_shape = (
            batch_size, context_size, embed_dim)
        act_output_shape = mha_output.shape
        _check_shape(exp_output_shape, act_output_shape)


def test_feedfwd_output_shape():
    TEST_CASES = [
        # tuple(embed dim, hidden size, number of hidden layers, batch size,
        # context size)
        (64, 128, 4, 10, 100),
        (128, 256, 8, 12, 200),
        (256, 512, 12, 1, 300),
        (512, 1024, 16, 3, 400)
    ]
    for test_case in TEST_CASES:
        embed_dim, hidden_size, num_hidden_layers, batch_size, context_size = \
            test_case
        ffwd = FeedFwd(embed_dim, hidden_size, num_hidden_layers)
        ffwd_input = torch.randn((batch_size, context_size, embed_dim))
        exp_output_shape = ffwd_input.shape
        ffwd_output = ffwd(ffwd_input)
        act_output_shape = ffwd_output.shape
        _check_shape(exp_output_shape, act_output_shape)


def test_attn_block_output_shape():
    TEST_CASES = [
        # tuple(batch size, embed dim, context size)
        (8, 32, 100),
        (16, 64, 200),
        (32, 128, 400),
        (64, 256, 800),
    ]
    for test_case in TEST_CASES:
        batch_size, embed_dim, context_size = test_case
        input_mask = torch.ones((batch_size, context_size))
        attn_block = AttnBlock(
            embed_dim, context_size, PLACEHOLDER_NUM_ATTN_HEADS,
            PLACEHOLDER_ATTN_HEAD_SIZE, PLACEHOLDER_HIDDEN_SIZE,
            PLACEHOLDER_NUM_HIDDEN_LAYERS)
        attn_block_input = torch.randn((batch_size, context_size, embed_dim))
        exp_output_shape = (batch_size, context_size, embed_dim)
        attn_block_output = attn_block(attn_block_input, input_mask)
        act_output_shape = attn_block_output.shape
        _check_shape(exp_output_shape, act_output_shape)


def test_sin_pos_embeds():
    TEST_CASES = [
        # tuple(context size, embed dim)
        (100, 32),
        (200, 64),
        (300, 128),
        (400, 256)
    ]
    for test_case in TEST_CASES:
        context_size, embed_dim = test_case
        pm = PoetronModel(
            PLACEHOLDER_VOCAB_SIZE, embed_dim, context_size,
            PLACEHOLDER_NUM_ATTN_HEADS, PLACEHOLDER_ATTN_HEAD_SIZE,
            PLACEHOLDER_HIDDEN_SIZE, PLACEHOLDER_NUM_HIDDEN_LAYERS,
            PLACEHOLDER_NUM_ATTN_BLOCKS)

        # check sinusoidal positional embedding shape
        spe = pm.sin_pos_embeds
        exp_spe_shape = (context_size, embed_dim)
        act_spe_shape = spe.shape
        _check_shape(exp_spe_shape, act_spe_shape)

        # check that generated sinusoidal positional embeddings are
        # deterministic
        for _ in range(100):
            new_spe = pm._get_sin_pos_embeds()
            assert torch.equal(spe, new_spe), \
                'Sinusoidal positional embeddings are not generated '\
                + 'deterministically'


def test_pos_embeds():
    TEST_CASES = [
        # tuple(context size, embed dim, number of zeros at the start of each
        # sequence in the batch)
        (100, 32, [10, 15, 20]),
        (200, 64, [20, 30, 40, 50]),
        (300, 128, [15, 30, 45, 60]),
        (400, 256, [20, 30, 50]),
        (100, 32, [0, 0, 10, 20]),
        (200, 64, [20, 30, 0, 0])
    ]
    for test_case in TEST_CASES:
        context_size, embed_dim, num_start_zeros_list = test_case
        pm = PoetronModel(
            PLACEHOLDER_VOCAB_SIZE, embed_dim, context_size,
            PLACEHOLDER_NUM_ATTN_HEADS, PLACEHOLDER_ATTN_HEAD_SIZE,
            PLACEHOLDER_HIDDEN_SIZE, PLACEHOLDER_NUM_HIDDEN_LAYERS,
            PLACEHOLDER_NUM_ATTN_BLOCKS)
        # create input mask
        input_mask = []
        for num_start_zeros in num_start_zeros_list:
            input_mask.append(
                [0] * num_start_zeros + [1] * (context_size - num_start_zeros))
        input_mask = torch.Tensor(input_mask)
        # get positional embeddings
        pos_embeds = pm._get_pos_embeds(input_mask)
        
        # check shape of positional embeddings
        exp_shape = (len(num_start_zeros_list), context_size, embed_dim)
        act_shape = pos_embeds.shape
        _check_shape(exp_shape, act_shape)

        # check where 0s/sinusoidal positional embeddings are used
        exp_pos_embeds = []
        for num_start_zeros in num_start_zeros_list:
            if num_start_zeros != 0:
                zeros = torch.zeros(num_start_zeros, embed_dim)
                sin_pos_embeds = pm.sin_pos_embeds[
                    :context_size - num_start_zeros]
                exp_pos_embeds.append(torch.cat([zeros, sin_pos_embeds], dim=0))
            else:
                exp_pos_embeds.append(pm.sin_pos_embeds)
        exp_pos_embeds = torch.stack(exp_pos_embeds, dim=0)
        assert torch.equal(pos_embeds, exp_pos_embeds), 'Incorrect positional '\
            + f'embeddings. Expected: {exp_pos_embeds}. Actual: {pos_embeds}'


def test_poetron_model_output_shape():
    TEST_CASES = [
        # tuple(batch size, context size, vocab size)
        (4, 100, 120),
        (8, 200, 240),
        (16, 300, 480),
        (32, 400, 600)
    ]
    for test_case in TEST_CASES:
        batch_size, context_size, vocab_size = test_case
        input_mask = torch.ones((batch_size, context_size))
        pm_input = torch.randint(
            low=0, high=vocab_size, size=(batch_size, context_size))
        exp_output_shape = (batch_size, context_size, vocab_size)
        pm = PoetronModel(
            vocab_size, PLACEHOLDER_EMBED_DIM, context_size,
            PLACEHOLDER_NUM_ATTN_HEADS, PLACEHOLDER_ATTN_HEAD_SIZE,
            PLACEHOLDER_HIDDEN_SIZE, PLACEHOLDER_NUM_HIDDEN_LAYERS,
            PLACEHOLDER_NUM_ATTN_BLOCKS)
        pm_output = pm(pm_input, input_mask)
        act_output_shape = pm_output.shape
        _check_shape(exp_output_shape, act_output_shape)


if __name__ == '__main__':
    # run tests
    tests = [
        test_get_dataset,
        test_get_tokenizer,
        test_get_batch,
        test_get_init_attn_pattern,
        test_scale_attn_pattern,
        test_apply_subseq_mask,
        test_apply_input_mask,
        test_resolve_neg_inf_rows,
        test_normalize_attn_pattern,
        test_self_attn_output_shape,
        test_get_concat_sah_outputs,
        test_multi_head_attn_output_shape,
        test_feedfwd_output_shape,
        test_attn_block_output_shape,
        test_sin_pos_embeds,
        test_pos_embeds,
        test_poetron_model_output_shape
    ]
    for test in tests:
        test()
        print(f'Test {test.__name__} passed.')

    # if this point is reached, no AssertionError was thrown in any test
    print('All tests passed!')
