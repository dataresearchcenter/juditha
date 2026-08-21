"""gRPC server exposing the read-only store surface.

No authentication and no TLS: this is meant to sit on a private network or
behind something that does terminate both. `Settings.rpc_host` therefore
defaults to `localhost`, not `0.0.0.0`.
"""

from concurrent import futures

import grpc
from anystore.logging import get_logger
from anystore.types import Uri

from juditha.rpc import juditha_pb2 as pb
from juditha.rpc import juditha_pb2_grpc as pb_grpc
from juditha.rpc.client import channel_options
from juditha.rpc.convert import to_pb_mention, to_pb_result
from juditha.settings import Settings
from juditha.store import BaseStore, get_store

log = get_logger(__name__)


class JudithaServicer(pb_grpc.JudithaServicer):
    def __init__(self, store: BaseStore) -> None:
        self.store = store

    def Search(
        self, request: pb.SearchRequest, context: grpc.ServicerContext
    ) -> pb.SearchResponse:
        result = self.store.search(
            request.q,
            threshold=request.threshold if request.HasField("threshold") else None,
            limit=request.limit if request.HasField("limit") else None,
            schemata=list(request.schemata) or None,
        )
        if result is None:
            # An unset `result` field is how "no match" crosses the wire.
            return pb.SearchResponse()
        return pb.SearchResponse(result=to_pb_result(result))

    def Extract(
        self, request: pb.ExtractRequest, context: grpc.ServicerContext
    ) -> pb.MentionsResponse:
        mentions = self.store.extract(request.text)
        return pb.MentionsResponse(mentions=[to_pb_mention(m) for m in mentions])

    def Percolate(
        self, request: pb.PercolateRequest, context: grpc.ServicerContext
    ) -> pb.MentionsResponse:
        mentions = self.store.percolate(request.text, slop=request.slop)
        return pb.MentionsResponse(mentions=[to_pb_mention(m) for m in mentions])


def make_server(
    store: BaseStore, host: str, port: int, workers: int
) -> tuple[grpc.Server, int]:
    """Build a started-but-not-serving server, and the port it bound.

    Split out of `serve` so tests can bind port 0 and read back the
    ephemeral port instead of guessing a free one.
    """
    # Warm the Aho-Corasick automaton before accepting traffic: the lazy
    # `Store.extractor` property is not synchronized, so two concurrent first
    # requests would each load the whole patterns file.
    store.extract("")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers), options=channel_options()
    )
    pb_grpc.add_JudithaServicer_to_server(JudithaServicer(store), server)
    return server, server.add_insecure_port(f"{host}:{port}")


def serve(
    uri: Uri | None = None,
    host: str | None = None,
    port: int | None = None,
    workers: int | None = None,
) -> None:
    """Serve the store at `uri` over gRPC until the process is terminated."""
    settings = Settings()
    host = host if host is not None else settings.rpc_host
    port = port if port is not None else settings.rpc_port
    workers = workers if workers is not None else settings.rpc_workers

    store = get_store(uri)
    server, bound = make_server(store, host, port, workers)
    server.start()
    log.info("🚀 Serving ...", host=host, port=bound, store=store.uri, workers=workers)
    server.wait_for_termination()
