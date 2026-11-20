"""gRPC client: a read-only store that proxies to a remote juditha server."""

from typing import Iterable

import grpc
from anystore.logging import get_logger

from juditha.model import Mention, Result
from juditha.rpc import juditha_pb2 as pb
from juditha.rpc import juditha_pb2_grpc as pb_grpc
from juditha.rpc.convert import from_pb_mention, from_pb_result
from juditha.settings import settings
from juditha.store import GRPC_SCHEME, BaseStore

log = get_logger(__name__)


def channel_options() -> list[tuple[str, int]]:
    """Shared client / server channel options.

    Raises gRPC's 4 MB default message limit: `extract` and `percolate` take
    whole documents as input and can return a mention per name on a
    multi-million-name corpus.
    """
    length = settings.rpc_max_message_length
    return [
        ("grpc.max_send_message_length", length),
        ("grpc.max_receive_message_length", length),
    ]


def make_search_request(
    q: str,
    threshold: float | None,
    limit: int | None,
    schemata: Iterable[str] | None,
    topics: Iterable[str] | None,
) -> pb.SearchRequest:
    request = pb.SearchRequest(
        q=q,
        schemata=sorted(schemata) if schemata else None,
        topics=sorted(topics) if topics else None,
    )
    # Leave the optional fields unset so the server falls back to its own
    # Settings defaults instead of ours.
    if threshold is not None:
        request.threshold = threshold
    if limit is not None:
        request.limit = limit
    return request


def unpack_search_response(response: pb.SearchResponse) -> Result | None:
    if not response.HasField("result"):
        return None
    return from_pb_result(response.result)


class ApiStore(BaseStore):
    """Read-only store backed by a remote juditha gRPC server.

    Returned by `juditha.store.get_store` for `grpc://host:port` URIs, so
    `lookup()` and the `extract` / `percolate` CLI commands work against a
    remote store without any call-site change.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.target = uri.removeprefix(GRPC_SCHEME)
        self.channel = grpc.insecure_channel(self.target, options=channel_options())
        self.stub = pb_grpc.JudithaStub(self.channel)
        log.info("👋 (remote)", store=self.uri)

    def search(
        self,
        q: str,
        threshold: float | None = None,
        limit: int | None = None,
        schemata: Iterable[str] | None = None,
        topics: Iterable[str] | None = None,
    ) -> Result | None:
        request = make_search_request(q, threshold, limit, schemata, topics)
        return unpack_search_response(self.stub.Search(request))

    def search_many(
        self,
        queries: Iterable[str],
        threshold: float | None = None,
        limit: int | None = None,
        schemata: Iterable[str] | None = None,
        topics: Iterable[str] | None = None,
    ) -> list[Result | None]:
        """Batched `search`: unique queries go out in chunks of
        `Settings.rpc_batch_size`, one `SearchMany` round trip per chunk."""
        queries = list(queries)
        unique = list(dict.fromkeys(queries))
        schemata = sorted(schemata) if schemata else None
        topics = sorted(topics) if topics else None
        size = max(1, settings.rpc_batch_size)
        found: dict[str, Result | None] = {}
        for i in range(0, len(unique), size):
            chunk = unique[i : i + size]
            request = pb.SearchManyRequest(
                requests=[
                    make_search_request(q, threshold, limit, schemata, topics)
                    for q in chunk
                ]
            )
            response = self.stub.SearchMany(request)
            for q, res in zip(chunk, response.responses, strict=True):
                found[q] = unpack_search_response(res)
        return [found[q] for q in queries]

    def extract(self, text: str) -> list[Mention]:
        response = self.stub.Extract(pb.ExtractRequest(text=text))
        return [from_pb_mention(m) for m in response.mentions]

    def percolate(self, text: str, slop: int = 0) -> list[Mention]:
        response = self.stub.Percolate(pb.PercolateRequest(text=text, slop=slop))
        return [from_pb_mention(m) for m in response.mentions]

    def close(self) -> None:
        self.channel.close()
