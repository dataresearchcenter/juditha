from anystore.io import logged_items, smart_stream
from anystore.logging import get_logger
from anystore.types import Uri
from ftmq import M, Query
from ftmq.io import smart_read_proxies
from ftmq.model.dataset import Catalog, Dataset
from ftmq.util import make_entity

from juditha.store import BuildStore, get_build_store

log = get_logger(__name__)


Q = Query(M(schemata="LegalEntity") | M(schema="Address"))


def load_proxies(
    uri: Uri, store: BuildStore | None = None, sync: bool | None = False
) -> None:
    store = store or get_build_store()
    entities = logged_items(
        Q.apply_iter(smart_read_proxies(uri)),
        "Load",
        item_name="Proxy",
        logger=log,
        uri=uri,
    )
    store.aggregator.load_entities(entities)
    if sync:
        store.build()


def load_dataset(
    uri: Uri, store: BuildStore | None = None, sync: bool | None = False
) -> None:
    store = store or get_build_store()
    dataset = Dataset._from_uri(uri)
    log.info(f"[{dataset.name}] Loading ...")
    entities = logged_items(
        Q.apply_iter(dataset.iterate()),
        "Load",
        item_name="Proxy",
        logger=log,
        dataset=dataset.name,
    )
    store.aggregator.load_entities(entities)
    if sync:
        store.build()


def load_catalog(
    uri: Uri, store: BuildStore | None = None, sync: bool | None = False
) -> None:
    store = store or get_build_store()
    catalog = Catalog._from_uri(uri)
    for dataset in catalog.datasets:
        if dataset.uri:
            load_dataset(dataset.uri, store)
    if sync:
        store.build()


def load_names(
    uri: Uri, store: BuildStore | None = None, schema: str | None = None
) -> None:
    store = store or get_build_store()
    schema = schema or "LegalEntity"
    with store.aggregator:
        for i, name in enumerate(
            logged_items(
                smart_stream(uri, mode="r"),
                "Load",
                item_name="Name",
                logger=log,
                uri=uri,
            )
        ):
            name = name.strip()
            entity = make_entity(
                {
                    "id": f"name-{i}",
                    "schema": schema,
                    "properties": {"name": [name]},
                }
            )
            store.aggregator.put(entity)
