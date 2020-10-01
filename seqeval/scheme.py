import abc
import enum


class Entity:

    def __init__(self, start, end, tag):
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


class Token(metaclass=abc.ABCMeta):
    allowed_prefix = None
    accepts_as_start = None
    accepts_as_inside = None
    accepts_as_end = None

    def __init__(self, token, suffix=False, delimiter='-'):
        self.token = token
        self.suffix = suffix
        self.delimiter = delimiter

    @property
    def prefix(self):
        prefix = self.token[-1] if self.suffix else self.token[0]
        return Prefix[prefix]

    @property
    def tag(self):
        tag = self.token[:-1] if self.suffix else self.token[1:]
        tag = tag.strip(self.delimiter) or '_'
        return tag

    def is_valid(self):
        if self.prefix not in self.allowed_prefix:
            allowed_prefixes = str(self.allowed_prefix).replace('Prefix.', '')
            message = 'Invalid token is found: {}. Allowed prefixes are: {}.'
            raise ValueError(message.format(self.token, allowed_prefixes))
        return True

    def is_start(self, prev):
        return self.check_pattern(prev, self.accepts_as_start)

    def is_inside(self, prev):
        if prev.tag != self.tag:
            return False
        return (prev.prefix, self.prefix) in self.accepts_as_inside

    def is_end(self, prev):
        return self.check_pattern(prev, self.accepts_as_end)

    def check_pattern(self, prev, pattern):
        return any(prev.prefix in b and self.prefix in a for b, a in pattern)


class IOB2(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    accepts_as_start = {
        (Prefix.ANY, Prefix.B)
    }
    accepts_as_inside = {
        (Prefix.B, Prefix.I),
        (Prefix.I, Prefix.I)
    }
    accepts_as_end = {
        (Prefix.I, Prefix.O),
        (Prefix.I, Prefix.B),
        (Prefix.B, Prefix.O),
        (Prefix.B, Prefix.B)
    }

    def is_end(self, prev):
        if prev.tag != self.tag:
            return True
        return super(IOB2, self).is_end(prev)


class IOE2(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E
    accepts_as_start = {
        (Prefix.O, Prefix.I),
        (Prefix.O, Prefix.E),
        (Prefix.E, Prefix.I),
        (Prefix.E, Prefix.E)
    }
    accepts_as_inside = {
        (Prefix.I, Prefix.E),
        (Prefix.I, Prefix.I)
    }
    accepts_as_end = {
        (Prefix.E, Prefix.ANY)
    }

    def is_start(self, prev):
        if prev.tag != self.tag:
            return True
        return super(IOE2, self).is_start(prev)


class IOBES(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B | Prefix.E | Prefix.S
    accepts_as_start = {
        (Prefix.ANY, Prefix.B),
        (Prefix.ANY, Prefix.S)
    }
    accepts_as_inside = {
        (Prefix.B, Prefix.I),
        (Prefix.I, Prefix.I),
        (Prefix.B, Prefix.E)
    }
    accepts_as_end = {
        (Prefix.S, Prefix.ANY),
        (Prefix.E, Prefix.ANY)
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
                    entities.append(entity)
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
        return len(self.tokens)

    def _is_end(self, pos):
        token = self.tokens[pos]
        prev = self.tokens[pos-1]
        return token.is_end(prev)


if __name__ == '__main__':
    sequences = [
        Tokens(['B-PER', 'I-PER', 'O', 'B-LOC'], IOB2),
        Tokens(['B-PER', 'I-ORG', 'B-ORG', 'B-LOC'], IOB2),
        Tokens(['I-PER', 'I-PER', 'I-ORG', 'I-LOC'], IOB2),
        #Tokens(['B-PER', 'E-PER', 'S-PER'], IOB2),
        Tokens(['B-PER', 'E-PER', 'S-PER'], IOBES),
        Tokens(['B-PER', 'I-PER', 'S-PER'], IOBES),
        Tokens(['I-PER', 'I-PER', 'I-ORG', 'I-LOC'], IOBES),
        Tokens(['B-PER', 'I-ORG', 'B-ORG', 'B-LOC'], IOBES),
        Tokens(['O', 'I', 'E', 'O', 'I', 'I', 'E', 'E', 'O', 'E', 'O'], IOE2)
    ]
    for tokens in sequences:
        print(tokens.entities)
    entity_set1 = {Entity(0, 1, 'PER'), Entity(2, 3, 'ORG')}
    entity_set2 = {Entity(0, 1, 'PER')}
    print(entity_set1 & entity_set2)
    print(entity_set1 - entity_set2)
