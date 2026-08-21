# Usage / gRPC api

`juditha` is an in-process library first, but the read-only half of the store (`search`, `extract`, `percolate`) can be served over gRPC. That lets many workers, or workers on other hosts, share one built corpus instead of each carrying their own copy.

!!! warning "No authentication"

    The api has no auth and no TLS. Anything that can reach the port can query the whole corpus. Bind it to a private network, or put something in front that terminates auth. `JUDITHA_RPC_HOST` therefore defaults to `localhost`, not `0.0.0.0`.

## Serving

Build a store the usual way, then serve it:

```bash
export JUDITHA_URI=/var/lib/juditha
juditha load-entities -i entities.ftm.json
juditha build
juditha serve --host 0.0.0.0 --port 50051
```

| Option | Default | Env var |
| --- | --- | --- |
| `--host` | `localhost` | `JUDITHA_RPC_HOST` |
| `--port` | `50051` | `JUDITHA_RPC_PORT` |
| `--workers` | `10` | `JUDITHA_RPC_WORKERS` |

The server opens the tantivy index and the Aho-Corasick automaton, but *not* the LevelDB aggregator, so other processes on the same host can keep reading the same store directory at the same time. See [Settings](../reference/settings.md#rpc) for the full list.

## Querying

Point `JUDITHA_URI` at the server instead of at a directory. Everything else stays the same:

```bash
export JUDITHA_URI=grpc://localhost:50051

juditha lookup "European Parliament"
juditha extract -i document.txt
juditha percolate -i document.txt --slop 1
```

From Python, `get_store()` returns an `ApiStore` for `grpc://` URIs and a local `Store` for everything else. Both implement the same `BaseStore` interface, so no call site changes:

```python
from juditha import get_store, lookup

store = get_store("grpc://localhost:50051")
store.search("European Parliament")            # -> Result | None
store.extract("The European Parliament met.")  # -> list[Mention]
store.percolate("The European Parliament met.")

# or via the env var, which `lookup()` picks up too
lookup("European Parliament")
```

Only the read methods cross the wire. `juditha build`, the `load-*` commands and `juditha iterate` are local-only and raise a `ValueError` against a `grpc://` URI, because they need the LevelDB aggregator.

## Docker

The published image runs the server as its entrypoint. Mount a built store at `/data`:

```bash
docker run -p 50051:50051 -v /var/lib/juditha:/data ghcr.io/dataresearchcenter/juditha
```

The image sets `JUDITHA_RPC_HOST=0.0.0.0` and `JUDITHA_URI=/data/juditha.db`.

## The wire contract

The service definition lives at [`juditha/rpc/juditha.proto`](https://github.com/dataresearchcenter/juditha/blob/main/juditha/rpc/juditha.proto), and the checked-in python stubs next to it are regenerated with `make proto`.

```proto
service Juditha {
  rpc Search    (SearchRequest)    returns (SearchResponse);
  rpc Extract   (ExtractRequest)   returns (MentionsResponse);
  rpc Percolate (PercolateRequest) returns (MentionsResponse);
}
```

Notes for clients in other languages:

- `SearchRequest.threshold` and `.limit` are `optional`. Leave them unset to get the server's configured defaults rather than sending your own.
- `SearchResponse.result` is unset when there is no match above the threshold. That is the wire form of `Store.search` returning `None`.
- `Result.common_schema` and `Result.caption` are computed by the server. The python client ignores them and recomputes locally via [rigour](https://github.com/opensanctions/rigour) and [followthemoney](https://followthemoney.tech); other clients can use them as-is. An empty `caption` means there was no name to pick.
- `Result.took` is measured server-side in milliseconds and does not include network time.
- `Mention.schema` matches the JSON field name of `juditha.model.Mention` (the python attribute is `schema_`).

Message size limits are raised from gRPC's 4 MB default to 64 MB on both ends, because `extract` and `percolate` take whole documents. Tune with `JUDITHA_RPC_MAX_MESSAGE_LENGTH`, and set it on the client too.
