from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Result(_message.Message):
    __slots__ = ("key", "names", "aliases", "countries", "schemata", "score", "query", "took", "common_schema", "caption")
    KEY_FIELD_NUMBER: _ClassVar[int]
    NAMES_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    COUNTRIES_FIELD_NUMBER: _ClassVar[int]
    SCHEMATA_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOOK_FIELD_NUMBER: _ClassVar[int]
    COMMON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    key: str
    names: _containers.RepeatedScalarFieldContainer[str]
    aliases: _containers.RepeatedScalarFieldContainer[str]
    countries: _containers.RepeatedScalarFieldContainer[str]
    schemata: _containers.RepeatedScalarFieldContainer[str]
    score: float
    query: str
    took: float
    common_schema: str
    caption: str
    def __init__(self, key: _Optional[str] = ..., names: _Optional[_Iterable[str]] = ..., aliases: _Optional[_Iterable[str]] = ..., countries: _Optional[_Iterable[str]] = ..., schemata: _Optional[_Iterable[str]] = ..., score: _Optional[float] = ..., query: _Optional[str] = ..., took: _Optional[float] = ..., common_schema: _Optional[str] = ..., caption: _Optional[str] = ...) -> None: ...

class Mention(_message.Message):
    __slots__ = ("text", "start", "end", "schema")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    text: str
    start: int
    end: int
    schema: str
    def __init__(self, text: _Optional[str] = ..., start: _Optional[int] = ..., end: _Optional[int] = ..., schema: _Optional[str] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("q", "threshold", "limit", "schemata")
    Q_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    SCHEMATA_FIELD_NUMBER: _ClassVar[int]
    q: str
    threshold: float
    limit: int
    schemata: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, q: _Optional[str] = ..., threshold: _Optional[float] = ..., limit: _Optional[int] = ..., schemata: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: Result
    def __init__(self, result: _Optional[_Union[Result, _Mapping]] = ...) -> None: ...

class ExtractRequest(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class PercolateRequest(_message.Message):
    __slots__ = ("text", "slop")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SLOP_FIELD_NUMBER: _ClassVar[int]
    text: str
    slop: int
    def __init__(self, text: _Optional[str] = ..., slop: _Optional[int] = ...) -> None: ...

class MentionsResponse(_message.Message):
    __slots__ = ("mentions",)
    MENTIONS_FIELD_NUMBER: _ClassVar[int]
    mentions: _containers.RepeatedCompositeFieldContainer[Mention]
    def __init__(self, mentions: _Optional[_Iterable[_Union[Mention, _Mapping]]] = ...) -> None: ...
