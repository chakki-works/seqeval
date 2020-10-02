import unittest

from seqeval.scheme import IOB2, IOBES, IOE2, Prefix, Tokens


class TestIOB2(unittest.TestCase):

    def test_prefix_I(self):
        token = IOB2('I')
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.I)

    def test_prefix_B(self):
        token = IOB2('B')
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.B)

    def test_prefix_O(self):
        token = IOB2('O')
        prefix = token.prefix
        self.assertEqual(prefix, Prefix.O)

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

    def test_tag(self):
        token = IOB2('I-MISC')
        tag = token.tag
        self.assertEqual(tag, 'MISC')

    def test_only_iob(self):
        token = IOB2('I')
        tag = token.tag
        self.assertEqual(tag, '_')

    def test_B_prefix_is_start(self):
        token = IOB2('B-ORG')
        prev = IOB2('I-ORG')
        is_start = token.is_start(prev)
        self.assertTrue(is_start)

    def test_I_prefix_is_not_start(self):
        token = IOB2('I-ORG')
        prev = IOB2('B-ORG')
        is_start = token.is_start(prev)
        self.assertFalse(is_start)

    def test_O_prefix_is_not_start(self):
        token = IOB2('O')
        prev = IOB2('B-ORG')
        is_start = token.is_start(prev)
        self.assertFalse(is_start)

    def test_I_after_B_is_inside(self):
        token = IOB2('I-ORG')
        prev = IOB2('B-ORG')
        is_inside = token.is_inside(prev)
        self.assertTrue(is_inside)

    def test_I_after_B_with_different_tag_is_not_inside(self):
        token = IOB2('I-ORG')
        prev = IOB2('B-MISC')
        is_inside = token.is_inside(prev)
        self.assertFalse(is_inside)

    def test_O_after_B_is_end(self):
        token = IOB2('O')
        prev = IOB2('B-MISC')
        is_end = token.is_end(prev)
        self.assertTrue(is_end)

    def test_O_after_I_is_end(self):
        token = IOB2('O')
        prev = IOB2('I-MISC')
        is_end = token.is_end(prev)
        self.assertTrue(is_end)

    def test_B_after_B_is_end(self):
        token = IOB2('B-MISC')
        prev = IOB2('B-MISC')
        is_end = token.is_end(prev)
        self.assertTrue(is_end)

    def test_B_after_I_is_end(self):
        token = IOB2('B-MISC')
        prev = IOB2('I-MISC')
        is_end = token.is_end(prev)
        self.assertTrue(is_end)


class TestTokens(unittest.TestCase):

    def entity_helper(self, tokens, scheme, expected):
        tokens = Tokens(tokens, scheme)
        entities = [e.to_tuple() for e in tokens.entities]
        self.assertEqual(entities, expected)

    def test_correct_iob2_tokens(self):
        tokens = ['B-PER', 'I-PER', 'O', 'B-LOC']
        expected = [('PER', 0, 2), ('LOC', 3, 4)]
        self.entity_helper(tokens, IOB2, expected)

    def test_iob2_with_wrong_token(self):
        tokens = ['B-PER', 'I-ORG', 'B-ORG', 'B-LOC']
        expected = [('PER', 0, 1), ('ORG', 2, 3), ('LOC', 3, 4)]
        self.entity_helper(tokens, IOB2, expected)

    def test_raise_exception_when_iobes_tokens_with_iob2_scheme(self):
        tokens = Tokens(['B-PER', 'E-PER', 'S-PER'], IOB2)
        with self.assertRaises(ValueError):
            tokens.entities

    def test_correct_iobes_tokens(self):
        tokens = ['B-PER', 'E-PER', 'S-PER']
        expected = [('PER', 0, 2), ('PER', 2, 3)]
        self.entity_helper(tokens, IOBES, expected)

    def test_iobes_with_wrong_token(self):
        tokens = ['B-PER', 'I-PER', 'S-PER']
        expected = [('PER', 2, 3)]
        self.entity_helper(tokens, IOBES, expected)

    def test_correct_ioe2_tokens(self):
        tokens = ['O', 'I', 'E', 'O', 'I', 'I', 'E', 'E', 'O', 'E', 'O']
        expected = [('_', 1, 3), ('_', 4, 7), ('_', 7, 8), ('_', 9, 10)]
        self.entity_helper(tokens, IOE2, expected)
