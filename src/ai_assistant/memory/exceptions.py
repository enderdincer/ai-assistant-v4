class MemoryError(Exception):
    pass


class MemoryConnectionError(MemoryError):
    pass


class MemoryQueryError(MemoryError):
    pass


class ImmutableFactError(MemoryError):
    pass


class FactNotFoundError(MemoryError):
    pass


class CollectionNotFoundError(MemoryError):
    pass


class EmbeddingError(MemoryError):
    pass
