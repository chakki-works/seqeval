import unittest

import pytest

from seqeval.scheme import IOB2, IOBES, IOE2, Prefix, Tokens, Token


class TestToken(unittest.TestCase):

    def test_extracts_prefix_I(self):
        token = Token('I-MISC')
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.I)

    def test_extracts_prefix_B(self):
        token = Token('B-MISC')
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.B)

    def test_extracts_prefix_O(self):
        token = Token('O')
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.O)

    def test_extracts_prefix_I_if_suffix_is_set(self):
        token = Token('MISC-I', suffix=True)
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.I)

    def test_extracts_prefix_B_if_suffix_is_set(self):
        token = Token('MISC-B', suffix=True)
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.B)

    def test_extracts_prefix_O_if_suffix_is_set(self):
        token = Token('O', suffix=True)
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.O)

    def test_extracts_tag(self):
        token = Token('I-MISC')
        tag = token.tag
        self.assertEqual(tag, 'MISC')

    def test_extracts_tag_if_suffix_is_set(self):
        token = Token('MISC-I', suffix=True)
        tag = token.tag
        self.assertEqual(tag, 'MISC')

    def test_extracts_tag_if_token_is_prefix_only(self):
        token = Token('I')
        tag = token.tag
        self.assertEqual(tag, '_')

    def test_extracts_underscore_as_tag_if_input_is_O(self):
        token = Token('O')
        tag = token.tag
        self.assertEqual(tag, '_')

    def test_extracts_tag_if_input_contains_two_underscore(self):
        token = Token('I-ORG-COMPANY')
        tag = token.tag
        self.assertEqual(tag, 'ORG-COMPANY')

    def test_extracts_tag_if_input_contains_two_underscore_with_suffix(self):
        token = Token('ORG-COMPANY-I', suffix=True)
        tag = token.tag
        self.assertEqual(tag, 'ORG-COMPANY')

    def test_extracts_non_ascii_tag(self):
        token = Token('I-組織')
        tag = token.tag
        self.assertEqual(tag, '組織')

    def test_raises_type_error_if_input_is_binary_string(self):
        token = Token('I-組織'.encode('utf-8'))
        with self.assertRaises(TypeError):
            tag = token.tag

    def test_raises_index_error_if_input_is_empty_string(self):
        token = Token('')
        with self.assertRaises(IndexError):
            prefix = token.prefix


class TestIOB2Token(unittest.TestCase):

    def test_invalid_prefix(self):
        token = IOB2('T')
        with self.assertRaises(KeyError):
            prefix = token.prefix

    def test_validate_invalid_prefix(self):
        token = IOB2('E')
        with self.assertRaises(ValueError):
            token.is_valid()

    def test_validate_valid_prefix(self):
        token = IOB2('B')
        is_valid = token.is_valid()
        self.assertTrue(is_valid)


@pytest.mark.parametrize(
    'prev, token, expected',
    [
        ('O', 'O', [False, False, False]),
        ('O', 'I-PER', [False, False, False]),
        ('O', 'B-PER', [True, False, False]),
        ('I-PER', 'O', [False, False, True]),
        ('I-PER', 'I-PER', [False, True, False]),
        ('I-PER', 'I-ORG', [False, False, True]),
        ('I-PER', 'B-PER', [True, False, True]),
        ('I-PER', 'B-ORG', [True, False, True]),
        ('B-PER', 'O', [False, False, True]),
        ('B-PER', 'I-PER', [False, True, False]),
        ('B-PER', 'I-ORG', [False, False, True]),
        ('B-PER', 'B-PER', [True, False, True]),
        ('B-PER', 'B-ORG', [True, False, True])
    ]
)
class TestIOB2Token:

    def test_start_inside_end(self, prev, token, expected):
        prev = IOB2(prev)
        token = IOB2(token)
        is_start = token.is_start(prev)
        is_inside = token.is_inside(prev)
        is_end = token.is_end(prev)
        actual = [is_start, is_inside, is_end]
        assert actual == expected


@pytest.mark.parametrize(
    'prev, token, expected',
    [
        ('O', 'O', [False, False, False]),
        ('O', 'I-PER', [True, False, False]),
        ('O', 'E-PER', [True, False, False]),
        ('I-PER', 'O', [False, False, False]),
        ('I-PER', 'I-PER', [False, True, False]),
        ('I-PER', 'I-ORG', [True, False, False]),
        ('I-PER', 'E-PER', [False, True, False]),
        ('I-PER', 'E-ORG', [True, False, False]),
        ('E-PER', 'O', [False, False, True]),
        ('E-PER', 'I-PER', [True, False, True]),
        ('E-PER', 'I-ORG', [True, False, True]),
        ('E-PER', 'E-PER', [True, False, True]),
        ('E-PER', 'E-ORG', [True, False, True])
    ]
)
class TestIOE2Token:

    def test_start_inside_end(self, prev, token, expected):
        prev = IOE2(prev)
        token = IOE2(token)
        is_start = token.is_start(prev)
        is_inside = token.is_inside(prev)
        is_end = token.is_end(prev)
        actual = [is_start, is_inside, is_end]
        assert actual == expected


@pytest.mark.parametrize(
    'prev, token, expected',
    [
        ('O', 'O', [False, False, False]),
        ('O', 'I-PER', [False, False, False]),
        ('O', 'B-PER', [True, False, False]),
        ('O', 'E-PER', [False, False, False]),
        ('O', 'S-PER', [True, False, False]),
        ('I-PER', 'O', [False, False, False]),
        ('I-PER', 'I-PER', [False, True, False]),
        ('I-PER', 'I-ORG', [False, False, False]),
        ('I-PER', 'B-PER', [True, False, False]),
        ('I-PER', 'E-PER', [False, True, False]),
        ('I-PER', 'E-ORG', [False, False, False]),
        ('I-PER', 'S-PER', [True, False, False]),
        ('B-PER', 'O',     [False, False, False]),
        ('B-PER', 'I-PER', [False, True, False]),
        ('B-PER', 'I-ORG', [False, False, False]),
        ('B-PER', 'E-PER', [False, True, False]),
        ('B-PER', 'E-ORG', [False, False, False]),
        ('B-PER', 'S-PER', [True, False, False]),
        ('E-PER', 'O', [False, False, True]),
        ('E-PER', 'I-PER', [False, False, True]),
        ('E-PER', 'B-PER', [True, False, True]),
        ('E-PER', 'E-PER', [False, False, True]),
        ('E-PER', 'S-PER', [True, False, True]),
        ('S-PER', 'O', [False, False, True]),
        ('S-PER', 'I-PER', [False, False, True]),
        ('S-PER', 'B-PER', [True, False, True]),
        ('S-PER', 'E-PER', [False, False, True]),
        ('S-PER', 'S-PER', [True, False, True])
    ]
)
class TestIOBESToken:

    def test_start_inside_end(self, prev, token, expected):
        prev = IOBES(prev)
        token = IOBES(token)
        is_start = token.is_start(prev)
        is_inside = token.is_inside(prev)
        is_end = token.is_end(prev)
        actual = [is_start, is_inside, is_end]
        assert actual == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['B-PER'], [('PER', 0, 1)]),
        (['I-PER'], []),
        (['O'], []),
        (['O', 'I-PER'], []),
        (['O', 'B-PER'], [('PER', 1, 2)]),
        (['I-PER', 'O'], []),
        (['I-PER', 'I-PER'], []),
        (['I-PER', 'I-ORG'], []),
        (['I-PER', 'B-PER'], [('PER', 1, 2)]),
        (['I-PER', 'B-ORG'], [('ORG', 1, 2)]),
        (['B-PER', 'O'], [('PER', 0, 1)]),
        (['B-PER', 'I-PER'], [('PER', 0, 2)]),
        (['B-PER', 'I-ORG'], [('PER', 0, 1)]),
        (['B-PER', 'B-PER'], [('PER', 0, 1), ('PER', 1, 2)]),
        (['B-PER', 'B-ORG'], [('PER', 0, 1), ('ORG', 1, 2)])
    ]
)
def test_iob2_tokens(tokens, expected):
    tokens = Tokens(tokens, IOB2)
    entities = tokens.entities
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['E-PER'], [('PER', 0, 1)]),
        (['I-PER'], []),
        (['O'], []),
        (['O', 'I-PER'], []),
        (['O', 'E-PER'], [('PER', 1, 2)]),
        (['I-PER', 'O'], []),
        (['I-PER', 'I-PER'], []),
        (['I-PER', 'I-ORG'], []),
        (['I-PER', 'E-PER'], [('PER', 0, 2)]),
        (['I-PER', 'E-ORG'], [('ORG', 1, 2)]),
        (['E-PER', 'O'], [('PER', 0, 1)]),
        (['E-PER', 'I-PER'], [('PER', 0, 1)]),
        (['E-PER', 'I-ORG'], [('PER', 0, 1)]),
        (['E-PER', 'E-PER'], [('PER', 0, 1), ('PER', 1, 2)]),
        (['E-PER', 'E-ORG'], [('PER', 0, 1), ('ORG', 1, 2)])
    ]
)
def test_ioe2_tokens(tokens, expected):
    tokens = Tokens(tokens, IOE2)
    entities = tokens.entities
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        (['O'], []),
        (['I-PER'], []),
        (['B-PER'], []),
        (['E-PER'], []),
        (['S-PER'], [('PER', 0, 1)]),
        (['O', 'O'], []),
        (['O', 'I-PER'], []),
        (['O', 'B-PER'], []),
        (['O', 'E-PER'], []),
        (['O', 'S-PER'], [('PER', 1, 2)]),
        (['I-PER', 'O'], []),
        (['I-PER', 'I-PER'], []),
        (['I-PER', 'I-ORG'], []),
        (['I-PER', 'B-PER'], []),
        (['I-PER', 'E-PER'], []),
        (['I-PER', 'E-ORG'], []),
        (['I-PER', 'S-PER'], [('PER', 1, 2)]),
        (['B-PER', 'O'], []),
        (['B-PER', 'I-PER'], []),
        (['B-PER', 'I-ORG'], []),
        (['B-PER', 'E-PER'], [('PER', 0, 2)]),
        (['B-PER', 'E-ORG'], []),
        (['B-PER', 'S-PER'], [('PER', 1, 2)]),
        (['E-PER', 'O'], []),
        (['E-PER', 'I-PER'], []),
        (['E-PER', 'B-PER'], []),
        (['E-PER', 'E-PER'], []),
        (['E-PER', 'S-PER'], [('PER', 1, 2)]),
        (['S-PER', 'O'], [('PER', 0, 1)]),
        (['S-PER', 'I-PER'], [('PER', 0, 1)]),
        (['S-PER', 'B-PER'], [('PER', 0, 1)]),
        (['S-PER', 'E-PER'], [('PER', 0, 1)]),
        (['S-PER', 'S-PER'], [('PER', 0, 1), ('PER', 1, 2)])
    ]
)
def test_iobes_tokens(tokens, expected):
    tokens = Tokens(tokens, IOBES)
    entities = tokens.entities
    assert entities == expected


class TestTokens:

    def test_raise_exception_when_iobes_tokens_with_iob2_scheme(self):
        tokens = Tokens(['B-PER', 'E-PER', 'S-PER'], IOB2)
        with pytest.raises(ValueError):
            entities = tokens.entities
