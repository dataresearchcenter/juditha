from anystore.io import logged_items, smart_stream
from ftmq import Query
from ftmq.io import smart_stream_proxies
from ftmq.model.dataset import Catalog, Dataset

from juditha.logging import get_logger
from juditha.store import Doc, Store, get_store

log = get_logger(__name__)


Q = Query().where(schema="LegalEntity", schema_include_descendants=True)


def load_proxies(uri: str, store: Store | None = None) -> None:
    with store or get_store() as store:
        for proxy in logged_items(
            Q.apply_iter(smart_stream_proxies(uri)),
            "Load",
            item_name="Proxy",
            logger=log,
            uri=uri,
        ):
            store.put(Doc.from_proxy(proxy))


def load_dataset(uri: str, store: Store | None = None) -> None:
    dataset = Dataset._from_uri(uri)
    log.info(f"[{dataset.name}] Loading ...")
    with store or get_store() as store:
        for proxy in logged_items(
            Q.apply_iter(dataset.iterate()),
            "Load",
            item_name="Proxy",
            logger=log,
            dataset=dataset.name,
        ):
            store.put(Doc.from_proxy(proxy))


def load_catalog(uri: str, store: Store | None = None) -> None:
    catalog = Catalog._from_uri(uri)
    for dataset in catalog.datasets:
        load_dataset(dataset.uri, store)


def load_names(uri: str, store: Store | None = None) -> None:
    with store or get_store() as store:
        for name in logged_items(
            smart_stream(uri), "Load", item_name="Name", logger=log, uri=uri
        ):
            name = name.strip()
            store.put(Doc(caption=name, names=[name]))
