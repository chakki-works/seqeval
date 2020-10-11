import pytest

from seqeval.scheme import IOB1, IOB2, IOBES, IOE1, IOE2, Prefix, Tokens, Token, auto_detect, Entity, Entities


def test_entity_repr():
    data = (0, 0, 0, 0)
    entity = Entity(*data)
    assert str(data) == str(entity)


@pytest.mark.parametrize(
    'data1, data2, expected',
    [
        ((0, 0, 0, 0), (0, 0, 0, 0), True),
        ((1, 0, 0, 0), (0, 0, 0, 0), False),
        ((0, 1, 0, 0), (0, 0, 0, 0), False),
        ((0, 0, 1, 0), (0, 0, 0, 0), False),
        ((0, 0, 0, 1), (0, 0, 0, 0), False)
    ]
)
def test_entity_equality(data1, data2, expected):
    entity1 = Entity(*data1)
    entity2 = Entity(*data2)
    is_equal = entity1 == entity2
    assert is_equal == expected


@pytest.mark.parametrize(
    'sequences, tag_name, expected',
    [
        ([['B-PER', 'B-ORG']], '', set()),
        ([['B-PER', 'B-ORG']], 'ORG', {Entity(0, 1, 2, 'ORG')}),
        ([['B-PER', 'B-ORG']], 'PER', {Entity(0, 0, 1, 'PER')})
    ]
)
def test_entities_filter(sequences, tag_name, expected):
    entities = Entities(sequences, IOB2)
    filtered = entities.filter(tag_name)
    assert filtered == expected


@pytest.mark.parametrize(
    'token, suffix, expected',
    [
        ('I-MISC', False, Prefix.I),
        ('B-MISC', False, Prefix.B),
        ('O', False, Prefix.O),
        ('MISC-I', True, Prefix.I),
        ('MISC-B', True, Prefix.B),
        ('O', True, Prefix.O)
    ]
)
def test_token_prefix(token, suffix, expected):
    token = Token(token, suffix=suffix)
    prefix = token.prefix
    assert prefix == expected


@pytest.mark.parametrize(
    'token, suffix, expected',
    [
        ('I-MISC', False, 'MISC'),
        ('MISC-I', True, 'MISC'),
        ('I', False, '_'),
        ('O', False, '_'),
        ('I-ORG-COMPANY', False, 'ORG-COMPANY'),
        ('ORG-COMPANY-I', True, 'ORG-COMPANY'),
        ('I-組織', False, '組織')
    ]
)
def test_token_tag(token, suffix, expected):
    token = Token(token, suffix=suffix)
    tag = token.tag
    assert tag == expected


def expects_start_inside_end_to_be_correct(prev, token, expected, scheme):
    prev = scheme(prev)
    token = scheme(token)
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
        ('O', 'B-PER', [False, False, False]),
        ('I-PER', 'O', [False, False, True]),
        ('I-PER', 'I-PER', [False, True, False]),
        ('I-PER', 'I-ORG', [True, False, True]),
        ('I-PER', 'B-PER', [True, False, True]),
        ('I-PER', 'B-ORG', [False, False, True]),
        ('B-PER', 'O', [False, False, True]),
        ('B-PER', 'I-PER', [True, True, False]),
        ('B-PER', 'I-ORG', [True, False, True]),
        ('B-PER', 'B-PER', [True, False, True]),
        ('B-PER', 'B-ORG', [False, False, False])
    ]
)
def test_iob1_start_inside_end(prev, token, expected):
    expects_start_inside_end_to_be_correct(prev, token, expected, IOB1)


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
def test_iob2_start_inside_end(prev, token, expected):
    expects_start_inside_end_to_be_correct(prev, token, expected, IOB2)


@pytest.mark.parametrize(
    'prev, token, expected',
    [
        ('O', 'O', [False, False, False]),
        ('O', 'I-PER', [True, False, False]),
        ('O', 'E-PER', [False, False, False]),
        ('I-PER', 'O', [False, False, True]),
        ('I-PER', 'I-PER', [False, True, False]),
        ('I-PER', 'I-ORG', [True, False, True]),
        ('I-PER', 'E-PER', [False, True, False]),
        ('I-PER', 'E-ORG', [False, False, True]),
        ('E-PER', 'O', [False, False, False]),
        ('E-PER', 'I-PER', [True, False, True]),
        ('E-PER', 'I-ORG', [True, False, False]),
        ('E-PER', 'E-PER', [True, False, True]),
        ('E-PER', 'E-ORG', [False, False, False])
    ]
)
def test_ioe1_start_inside_end(prev, token, expected):
    expects_start_inside_end_to_be_correct(prev, token, expected, IOE1)


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
def test_ioe2_start_inside_end(prev, token, expected):
    expects_start_inside_end_to_be_correct(prev, token, expected, IOE2)


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
def test_iobes_start_inside_end(prev, token, expected):
    expects_start_inside_end_to_be_correct(prev, token, expected, IOBES)


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['B-PER'], []),
        (['I-PER'], [('PER', 0, 1)]),
        (['O'], []),
        (['O', 'I-PER'], [('PER', 1, 2)]),
        (['O', 'B-PER'], []),
        (['I-PER', 'O'], [('PER', 0, 1)]),
        (['I-PER', 'I-PER'], [('PER', 0, 2)]),
        (['I-PER', 'I-ORG'], [('PER', 0, 1), ('ORG', 1, 2)]),
        (['I-PER', 'B-PER'], [('PER', 0, 1), ('PER', 1, 2)]),
        (['I-PER', 'B-ORG'], [('PER', 0, 1)]),
        (['B-PER', 'O'], []),
        (['B-PER', 'I-PER'], [('PER', 1, 2)]),
        (['B-PER', 'I-ORG'], [('ORG', 1, 2)]),
        (['B-PER', 'B-PER'], [('PER', 1, 2)]),
        (['B-PER', 'B-ORG'], [])
    ]
)
def test_iob1_tokens(tokens, expected):
    tokens = Tokens(tokens, IOB1)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['B'], []),
        (['I'], [('_', 0, 1)]),
        (['O'], []),
        (['O', 'O'], []),
        (['O', 'I'], [('_', 1, 2)]),
        (['O', 'B'], []),
        (['I', 'O'], [('_', 0, 1)]),
        (['I', 'I'], [('_', 0, 2)]),
        (['I', 'B'], [('_', 0, 1), ('_', 1, 2)]),
        (['B', 'O'], []),
        (['B', 'I'], [('_', 1, 2)]),
        (['B', 'B'], [('_', 1, 2)])
    ]
)
def test_iob1_tokens_without_tag(tokens, expected):
    tokens = Tokens(tokens, IOB1)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


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
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['B'], [('_', 0, 1)]),
        (['I'], []),
        (['O'], []),
        (['O', 'O'], []),
        (['O', 'I'], []),
        (['O', 'B'], [('_', 1, 2)]),
        (['I', 'O'], []),
        (['I', 'I'], []),
        (['I', 'B'], [('_', 1, 2)]),
        (['B', 'O'], [('_', 0, 1)]),
        (['B', 'I'], [('_', 0, 2)]),
        (['B', 'B'], [('_', 0, 1), ('_', 1, 2)])
    ]
)
def test_iob2_tokens_without_tag(tokens, expected):
    tokens = Tokens(tokens, IOB2)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['E-PER'], []),
        (['I-PER'], [('PER', 0, 1)]),
        (['O'], []),
        (['O', 'I-PER'], [('PER', 1, 2)]),
        (['O', 'E-PER'], []),
        (['I-PER', 'O'], [('PER', 0, 1)]),
        (['I-PER', 'I-PER'], [('PER', 0, 2)]),
        (['I-PER', 'I-ORG'], [('PER', 0, 1), ('ORG', 1, 2)]),
        # (['I-PER', 'E-PER'], [('PER', 0, 1)]),
        (['I-PER', 'E-ORG'], [('PER', 0, 1)]),
        (['E-PER', 'O'], []),
        (['E-PER', 'I-PER'], [('PER', 1, 2)]),
        (['E-PER', 'I-ORG'], [('ORG', 1, 2)]),
        (['E-PER', 'E-PER'], []),
        (['E-PER', 'E-ORG'], [])
    ]
)
def test_ioe1_tokens(tokens, expected):
    tokens = Tokens(tokens, IOE1)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['E'], []),
        (['I'], [('_', 0, 1)]),
        (['O'], []),
        (['O', 'O'], []),
        (['O', 'I'], [('_', 1, 2)]),
        (['O', 'E'], []),
        (['I', 'O'], [('_', 0, 1)]),
        (['I', 'I'], [('_', 0, 2)]),
        # (['I', 'E'], [('_', 0, 1)]),
        (['E', 'O'], []),
        (['E', 'I'], [('_', 1, 2)]),
        (['E', 'E'], [])
    ]
)
def test_ioe1_tokens_without_tag(tokens, expected):
    tokens = Tokens(tokens, IOE1)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
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
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        ([], []),
        (['E'], [('_', 0, 1)]),
        (['I'], []),
        (['O'], []),
        (['O', 'O'], []),
        (['O', 'I'], []),
        (['O', 'E'], [('_', 1, 2)]),
        (['I', 'O'], []),
        (['I', 'I'], []),
        (['I', 'E'], [('_', 0, 2)]),
        (['E', 'O'], [('_', 0, 1)]),
        (['E', 'I'], [('_', 0, 1)]),
        (['E', 'E'], [('_', 0, 1), ('_', 1, 2)])
    ]
)
def test_ioe2_tokens_without_tag(tokens, expected):
    tokens = Tokens(tokens, IOE2)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
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
        (['B-PER', 'B-PER'], []),
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
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


@pytest.mark.parametrize(
    'tokens, expected',
    [
        (['O'], []),
        (['I'], []),
        (['B'], []),
        (['E'], []),
        (['S'], [('_', 0, 1)]),
        (['O', 'O'], []),
        (['O', 'I'], []),
        (['O', 'B'], []),
        (['O', 'E'], []),
        (['O', 'S'], [('_', 1, 2)]),
        (['I', 'O'], []),
        (['I', 'I'], []),
        (['I', 'B'], []),
        (['I', 'E'], []),
        (['I', 'S'], [('_', 1, 2)]),
        (['B', 'O'], []),
        (['B', 'I'], []),
        (['B', 'B'], []),
        (['B', 'E'], [('_', 0, 2)]),
        (['B', 'S'], [('_', 1, 2)]),
        (['E', 'O'], []),
        (['E', 'I'], []),
        (['E', 'B'], []),
        (['E', 'E'], []),
        (['E', 'S'], [('_', 1, 2)]),
        (['S', 'O'], [('_', 0, 1)]),
        (['S', 'I'], [('_', 0, 1)]),
        (['S', 'B'], [('_', 0, 1)]),
        (['S', 'E'], [('_', 0, 1)]),
        (['S', 'S'], [('_', 0, 1), ('_', 1, 2)])
    ]
)
def test_iobes_tokens_without_tag(tokens, expected):
    tokens = Tokens(tokens, IOBES)
    entities = [entity.to_tuple()[1:] for entity in tokens.entities]
    assert entities == expected


class TestToken:

    def test_raises_type_error_if_input_is_binary_string(self):
        token = Token('I-組織'.encode('utf-8'))
        with pytest.raises(TypeError):
            tag = token.tag

    def test_raises_index_error_if_input_is_empty_string(self):
        token = Token('')
        with pytest.raises(IndexError):
            prefix = token.prefix


class TestIOB2Token:

    def test_invalid_prefix(self):
        token = IOB2('T')
        with pytest.raises(KeyError):
            prefix = token.prefix


@pytest.mark.parametrize(
    'token, scheme',
    [
        ('I', IOB1), ('O', IOB1), ('B', IOB1),
        ('I', IOB2), ('O', IOB2), ('B', IOB2),
        ('I', IOE1), ('O', IOE1), ('E', IOE1),
        ('I', IOE2), ('O', IOE2), ('E', IOE2),
        ('I', IOBES), ('O', IOBES), ('B', IOBES), ('E', IOBES), ('S', IOBES)
    ]
)
def test_valid_prefix(token, scheme):
    token = scheme(token)
    is_valid = token.is_valid
    assert is_valid


@pytest.mark.parametrize(
    'token, scheme',
    [
        ('E', IOB1), ('S', IOB1),
        ('E', IOB2), ('S', IOB2),
        ('B', IOE1), ('S', IOE1),
        ('B', IOE2), ('S', IOE2)
    ]
)
def test_invalid_prefix(token, scheme):
    token = scheme(token)
    with pytest.raises(ValueError):
        token.is_valid()


class TestTokens:

    def test_raise_exception_when_iobes_tokens_with_iob2_scheme(self):
        tokens = Tokens(['B-PER', 'E-PER', 'S-PER'], IOB2)
        with pytest.raises(ValueError):
            entities = tokens.entities


class TestAutoDetect:

    @pytest.mark.parametrize(
        'sequences, expected',
        [
            ([['B', 'I', 'O']], IOB2),
            ([['B', 'I']], IOB2),
            ([['B', 'O']], IOB2),
            ([['B']], IOB2),
            ([['I', 'O', 'E']], IOE2),
            ([['I', 'E']], IOE2),
            ([['E', 'O']], IOE2),
            ([['E']], IOE2),
            ([['I', 'O', 'B', 'E', 'S']], IOBES),
            ([['I', 'B', 'E', 'S']], IOBES),
            ([['I', 'O', 'B', 'E']], IOBES),
            ([['O', 'B', 'E', 'S']], IOBES),
            ([['I', 'B', 'E']], IOBES),
            ([['B', 'E', 'S']], IOBES),
            ([['O', 'B', 'E']], IOBES),
            ([['B', 'E']], IOBES),
            ([['S']], IOBES)
         ]
    )
    def test_valid_scheme(self, sequences, expected):
        scheme = auto_detect(sequences)
        assert scheme == expected

    @pytest.mark.parametrize(
        'sequences, expected',
        [
            ([['I', 'O']], IOB2),
            ([['H']], IOB2)
        ]
    )
    def test_invalid_scheme(self, sequences, expected):
        with pytest.raises(ValueError):
            scheme = auto_detect(sequences)

