from poetron import Poetron, INT_DATA_TYPE
from poetron_model import SelfAttnHead
import torch


# placeholder constants
PLACEHOLDER_EMBED_DIM = 20
PLACEHOLDER_ATTN_HEAD_SIZE = 10


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
        int_tensors = tokenizer.encode([text])

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

    # check that batch of encoded inputs are padded to the max length
    int_tensors = tokenizer.encode(
        list(map(lambda t: t[0], texts)), pad_or_truncate=True)
    # check tensor data types
    for int_tensor in int_tensors:
        assert int_tensor.dtype is INT_DATA_TYPE, 'Incorrect data type '\
            + f'for tokens tensor. Expected: {INT_DATA_TYPE}. Actual: '\
            + f'{int_tensor.dtype}'
    # check shape of encoded inputs and number of padding tokens
    int_tensors = torch.stack(int_tensors, dim=0)
    exp_shape = (len(texts), tokenizer.max_length)
    assert int_tensors.shape == exp_shape, 'Incorrect shape of encoding '\
        + f'output. Expected: {exp_shape}. Actual: {int_tensors.shape}'
    for i in range(len(int_tensors)):
        int_tensor = int_tensors[i]
        padding_tokens = set(
            int_tensor[:tokenizer.max_length - texts[i][1]].tolist())
        exp_padding_tokens = {tokenizer.token_to_int['<padding>']}
        assert padding_tokens == exp_padding_tokens, 'Incorrect padding '\
            f'tokens. Expected: {exp_padding_tokens}. Actual: {padding_tokens}'


def test_get_batch():
    # create Poetron instance, get dataset and tokenizer
    p = Poetron()
    p._get_dataset()
    p._get_tokenizer()

    # check batch inputs and next tokens have expected shape and data type
    batch_sizes = [2, 4, 8, 16, 32, 64]
    for batch_size in batch_sizes:
        batch_inputs, batch_next_tokens = p._get_batch(batch_size)

        # check shape of batch of inputs
        exp_shape = (batch_size, p.context_size)
        act_shape = batch_inputs.shape
        assert exp_shape == act_shape, 'Incorrect shape of batch of input '\
            + f'tokens. Expected: {exp_shape}. Actual: {act_shape}'

        # check that all token integers are present in the tokenizer's mapping
        token_ints = set(batch_inputs.flatten().tolist())
        mapping_ints = set(p.tokenizer.int_to_token.keys())
        other_ints = mapping_ints.difference(token_ints)
        assert len(other_ints) != 0, 'Batch of input tokens contains '\
            + 'integers outside of tokenizer\'s int to token mapping'

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
        input_mask = torch.ones((context_size,))
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, attn_head_size, context_size, input_mask)
        act_attn_pattern = sah._get_init_attn_pattern(q, k)
        assert torch.equal(exp_attn_pattern, act_attn_pattern),\
            'Expected and actual attention patterns don\'t match for test '\
            + f'case {i + 1}. Expected: {exp_attn_pattern}, Actual: {act_attn_pattern}'


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
        input_mask = torch.ones((context_size,))
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size,
            input_mask)
        act_scaled_attn_pattern = sah._scale_attn_pattern(
            attn_pattern, scaling_factor)
        assert torch.equal(exp_scaled_attn_pattern, act_scaled_attn_pattern),\
            'Expected and actual attention patterns don\'t match for test '\
            + f'case {i + 1}. Expected: {exp_scaled_attn_pattern}, Actual: '\
            + f'{act_scaled_attn_pattern}'


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
        input_mask = torch.ones((context_size,))
        sah = SelfAttnHead(
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size,
            input_mask)
        act_masked_attn_pattern = sah._apply_subseq_mask(attn_pattern)
        assert torch.equal(exp_masked_attn_pattern, act_masked_attn_pattern),\
            'Expected and actual attention patterns don\'t match for test '\
            + f'case {i + 1}. Expected: {exp_masked_attn_pattern}, Actual: '\
            + f'{act_masked_attn_pattern}'


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
            PLACEHOLDER_EMBED_DIM, PLACEHOLDER_ATTN_HEAD_SIZE, context_size,
            input_mask)
        act_masked_attn_pattern = sah._apply_input_mask(attn_pattern)
        assert torch.equal(exp_masked_attn_pattern, act_masked_attn_pattern),\
            'Expected and actual attention patterns don\'t match for test '\
            + f'case {i + 1}. Expected: {exp_masked_attn_pattern}, Actual: '\
            + f'{act_masked_attn_pattern}'


def test_resolve_neg_inf_rows():
    TEST_CASES = [
        # tuple(attention pattern, expected attention pattern after resolving
        # rows with all -inf values)
    ]
    pass


def test_normalize_attn_pattern():
    TEST_CASES = [
        # tuple(attention pattern, expected normalized attention pattern)
    ]
    pass


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
        test_normalize_attn_pattern
    ]
    for test in tests:
        test()
        print(f'Test {test.__name__} passed.')

    # if this point is reached, no AssertionError was thrown in any test
    print('All tests passed!')
