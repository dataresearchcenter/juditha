from anystore.io import logged_items, smart_stream
from anystore.logging import get_logger
from anystore.types import Uri
from ftmq import Query
from ftmq.io import smart_read_proxies
from ftmq.model.dataset import Catalog, Dataset

from juditha.store import Doc, Store, get_store

log = get_logger(__name__)


SCHEMATA = ["LegalEntity", "Organization", "PublicBody", "Company", "Person", "Address"]
Q = Query().where(schema__in=SCHEMATA)


def load_proxies(
    uri: Uri, store: Store | None = None, sync: bool | None = False
) -> None:
    store = store or get_store()
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
    uri: Uri, store: Store | None = None, sync: bool | None = False
) -> None:
    store = store or get_store()
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
    uri: Uri, store: Store | None = None, sync: bool | None = False
) -> None:
    store = store or get_store()
    catalog = Catalog._from_uri(uri)
    for dataset in catalog.datasets:
        if dataset.uri:
            load_dataset(dataset.uri, store)
    if sync:
        store.build()


def load_names(uri: Uri, store: Store | None = None, schema: str | None = None) -> None:
    schemata = {schema} if schema else set()
    with store or get_store() as store:
        for name in logged_items(
            smart_stream(uri), "Load", item_name="Name", logger=log, uri=uri
        ):
            name = name.strip()
            store.put(Doc(caption=name, names={name}, schemata=schemata))
