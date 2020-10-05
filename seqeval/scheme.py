import enum


class Entity:

    def __init__(self, start: int, end: int, tag: str):
        self.start = start
        self.end = end
        self.tag = tag

    def __repr__(self):
        return '({}, {}, {})'.format(self.tag, self.start, self.end)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.tag == other.tag

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        return self.tag, self.start, self.end


class Prefix(enum.Flag):
    I = enum.auto()
    O = enum.auto()
    B = enum.auto()
    E = enum.auto()
    S = enum.auto()
    ANY = I | O | B | E | S


class Tag(enum.Flag):
    SAME = enum.auto()
    DIFF = enum.auto()
    ANY = SAME | DIFF


class Token:
    allowed_prefix = None
    start_patterns = None
    inside_patterns = None
    end_patterns = None

    def __init__(self, token: str, suffix: bool = False, delimiter: str = '-'):
        self.token = token
        self.suffix = suffix
        self.delimiter = delimiter

    def __repr__(self):
        return self.token

    @property
    def prefix(self):
        """Extracts a prefix from the token."""
        prefix = self.token[-1] if self.suffix else self.token[0]
        return Prefix[prefix]

    @property
    def tag(self):
        """Extracts a tag from the token."""
        tag = self.token[:-1] if self.suffix else self.token[1:]
        tag = tag.strip(self.delimiter) or '_'
        return tag

    def is_valid(self):
        """Check whether the prefix is allowed or not."""
        if self.prefix not in self.allowed_prefix:
            allowed_prefixes = str(self.allowed_prefix).replace('Prefix.', '')
            message = 'Invalid token is found: {}. Allowed prefixes are: {}.'
            raise ValueError(message.format(self.token, allowed_prefixes))
        return True

    def is_start(self, prev: 'Token'):
        """The current token is the start of chunk."""
        return self.check_patterns(prev, self.start_patterns)

    def is_inside(self, prev: 'Token'):
        """The current token is inside of chunk."""
        return self.check_patterns(prev, self.inside_patterns)

    def is_end(self, prev: 'Token'):
        """The previous token is the end of chunk."""
        return self.check_patterns(prev, self.end_patterns)

    def check_tag(self, prev, cond):
        """Check whether the tag pattern is matched."""
        if cond == Tag.ANY:
            return True
        if prev.tag == self.tag and cond == Tag.SAME:
            return True
        if prev.tag != self.tag and cond == Tag.DIFF:
            return True
        return False

    def check_patterns(self, prev, patterns):
        """Check whether the prefix patterns are matched."""
        for prev_prefix, current_prefix, tag_cond in patterns:
            if prev.prefix in prev_prefix and self.prefix in current_prefix and self.check_tag(prev, tag_cond):
                return True
        return False


class IOB1(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.B, Tag.SAME),
        (Prefix.B, Prefix.B, Tag.SAME)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.B, Tag.ANY),
        (Prefix.B, Prefix.O, Tag.ANY),
        (Prefix.B, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.B, Tag.SAME)
    }


class IOE1(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.E, Prefix.I, Tag.ANY),
        (Prefix.E, Prefix.E, Tag.SAME)
    }
    inside_patterns = {
        (Prefix.I, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.E, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.E, Tag.DIFF),
        (Prefix.E, Prefix.I, Tag.SAME),
        (Prefix.E, Prefix.E, Tag.SAME)
    }


class IOB2(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    start_patterns = {
        (Prefix.ANY, Prefix.B, Tag.ANY)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.B, Tag.ANY),
        (Prefix.B, Prefix.O, Tag.ANY),
        (Prefix.B, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.B, Tag.ANY)
    }


class IOE2(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.O, Prefix.E, Tag.ANY),
        (Prefix.E, Prefix.I, Tag.ANY),
        (Prefix.E, Prefix.E, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.E, Tag.DIFF)
    }
    inside_patterns = {
        (Prefix.I, Prefix.E, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.E, Prefix.ANY, Tag.ANY)
    }


class IOBES(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B | Prefix.E | Prefix.S
    start_patterns = {
        (Prefix.ANY, Prefix.B, Tag.ANY),
        (Prefix.ANY, Prefix.S, Tag.ANY)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.B, Prefix.E, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.E, Tag.SAME)
    }
    end_patterns = {
        (Prefix.S, Prefix.ANY, Tag.ANY),
        (Prefix.E, Prefix.ANY, Tag.ANY)
    }


class Tokens:

    def __init__(self, tokens, token_class):
        self.tokens = [token_class(token) for token in tokens] + [token_class('O')]
        self.token_class = token_class

    @property
    def entities(self):
        """Extract entities from tokens.

        Returns:
            list: list of Entity.

        Example:
            >>> tokens = Tokens(['B-PER', 'I-PER', 'O', 'B-LOC'], IOB2)
            >>> tokens.entities
            [('PER', 0, 2), ('LOC', 3, 4)]
        """
        i = 0
        prev = self.token_class('O')
        entities = []
        while i < len(self.tokens):
            token = self.tokens[i]
            token.is_valid()
            if token.is_start(prev):
                end = self._forward(start=i + 1, prev=token)
                if self._is_end(end):
                    entity = Entity(start=i, end=end, tag=token.tag)
                    entities.append(entity.to_tuple())
                i = end
            else:
                i += 1
            prev = self.tokens[i - 1]
        return entities

    def _forward(self, start, prev):
        for i, token in enumerate(self.tokens[start:], start):
            if token.is_inside(prev):
                prev = token
            else:
                return i
        return len(self.tokens) - 2

    def _is_end(self, i):
        token = self.tokens[i]
        prev = self.tokens[i - 1]
        return token.is_end(prev)
