import unittest

from seqeval.scheme import IOB2, Prefix


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
