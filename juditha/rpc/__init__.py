"""gRPC transport for the read-only store surface.

`juditha.proto` is the wire contract; the `juditha_pb2*` modules next to it
are generated from it by `make proto` and are checked in. `client.ApiStore`
is a `BaseStore` implementation that `juditha.store.get_store` returns for
`grpc://` URIs; `server.serve` is the other end.
"""
