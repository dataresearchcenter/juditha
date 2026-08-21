"""Conversion between the pydantic models in `juditha.model` and their
protobuf mirrors in `juditha.proto`.

Explicit rather than generic (`MessageToDict` / `ParseDict`) so the two
schemas stay visibly coupled and mypy can check the mapping.
"""

from juditha.model import Mention, Result
from juditha.rpc import juditha_pb2 as pb


def to_pb_result(result: Result) -> pb.Result:
    return pb.Result(
        key=result.key,
        names=sorted(result.names),
        aliases=sorted(result.aliases),
        countries=sorted(result.countries),
        schemata=sorted(result.schemata),
        score=result.score,
        query=result.query,
        took=result.took,
        # Output only: both are pydantic computed fields, so `from_pb_result`
        # drops them and lets pydantic recompute. They travel for the benefit
        # of clients that have no rigour / followthemoney to compute them.
        common_schema=result.common_schema,
        caption=result.caption or "",
    )


def from_pb_result(msg: pb.Result) -> Result:
    return Result(
        key=msg.key,
        names=set(msg.names),
        aliases=set(msg.aliases),
        countries=set(msg.countries),
        schemata=set(msg.schemata),
        score=msg.score,
        query=msg.query,
        took=msg.took,
    )


def to_pb_mention(mention: Mention) -> pb.Mention:
    return pb.Mention(
        text=mention.text,
        start=mention.start,
        end=mention.end,
        schema=mention.schema_,
    )


def from_pb_mention(msg: pb.Mention) -> Mention:
    return Mention(text=msg.text, start=msg.start, end=msg.end, schema=msg.schema)
