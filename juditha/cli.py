from typing import Annotated, Optional

import typer
from anystore.cli import ErrorHandler
from anystore.io import smart_write_models
from anystore.logging import configure_logging
from rich import print

from juditha import __version__, io
from juditha.enricher import dbpedia
from juditha.settings import Settings
from juditha.store import get_store, lookup

settings = Settings()

cli = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=settings.debug)


@cli.callback(invoke_without_command=True)
def cli_juditha(
    version: Annotated[Optional[bool], typer.Option(..., help="Show version")] = False,
    settings: Annotated[
        Optional[bool], typer.Option(..., help="Show current settings")
    ] = False,
):
    if version:
        print(__version__)
        raise typer.Exit()
    if settings:
        print(settings)
        raise typer.Exit()
    configure_logging()


@cli.command()
def load_entities(
    uri: Annotated[str, typer.Option("-i", help="Input uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_proxies(uri)


@cli.command()
def load_names(
    uri: Annotated[str, typer.Option("-i", help="Input uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_names(uri)


@cli.command()
def load_dataset(
    uri: Annotated[str, typer.Option("-i", help="Dataset uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_dataset(uri)


@cli.command()
def load_catalog(
    uri: Annotated[str, typer.Option("-i", help="Catalog uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_catalog(uri)


@cli.command("lookup")
def cli_lookup(
    value: str,
    threshold: Annotated[
        float, typer.Option(..., help="Fuzzy threshold")
    ] = settings.fuzzy_threshold,
):
    with ErrorHandler():
        result = lookup(value, threshold=threshold)
        if result is not None:
            print(result)
        else:
            print("[red]not found[/red]")


@cli.command("build")
def cli_build():
    with ErrorHandler():
        store = get_store()
        store.build()


@cli.command("load-dbpedia")
def cli_load_dbpedia(
    in_uri: Annotated[str, typer.Option("-i", help="Input uri, default stdin")] = "-",
    out_uri: Annotated[
        str, typer.Option("-o", help="Output uri, default stdout")
    ] = "-",
):
    """Generate Person entities from dbpedia persondata dumps"""
    smart_write_models(out_uri, dbpedia.stream_dbpedia(in_uri))
