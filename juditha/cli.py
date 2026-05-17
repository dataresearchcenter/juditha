from typing import Annotated, Optional

import typer
from anystore.cli import ErrorHandler
from anystore.io import smart_read, smart_write_models
from anystore.logging import configure_logging
from rich import print

from juditha import __version__, io
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
        print(Settings())
        raise typer.Exit()
    configure_logging()


@cli.command("load-entities")
def cli_load_entities(
    uri: Annotated[str, typer.Option("-i", help="Input uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_proxies(uri)


@cli.command("load-names")
def cli_load_names(
    uri: Annotated[str, typer.Option("-i", help="Input uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_names(uri)


@cli.command("load-dataset")
def cli_load_dataset(
    uri: Annotated[str, typer.Option("-i", help="Dataset uri, default stdin")] = "-",
):
    with ErrorHandler():
        io.load_dataset(uri)


@cli.command("load-catalog")
def cli_load_catalog(
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


@cli.command("iterate")
def cli_iterate(
    output_uri: Annotated[
        str, typer.Option("-o", help="Output uri, default stdout")
    ] = "-",
):
    """Iterate through names.db"""
    with ErrorHandler():
        store = get_store()
        smart_write_models(output_uri, store.aggregator.iterate())


@cli.command("extract")
def cli_extract(
    input_uri: Annotated[
        str, typer.Option("-i", help="Input uri, default stdin")
    ] = "-",
    output_uri: Annotated[
        str, typer.Option("-o", help="Output uri, default stdout")
    ] = "-",
):
    """Extract known entity mentions from text."""
    with ErrorHandler():
        store = get_store()
        text = smart_read(input_uri, mode="r")
        mentions = store.extract(text)
        smart_write_models(output_uri, iter(mentions))


@cli.command("percolate")
def cli_percolate(
    input_uri: Annotated[
        str, typer.Option("-i", help="Input uri, default stdin")
    ] = "-",
    output_uri: Annotated[
        str, typer.Option("-o", help="Output uri, default stdout")
    ] = "-",
    slop: Annotated[
        int,
        typer.Option("--slop", help="Allowed intervening tokens between name parts"),
    ] = 0,
):
    """Percolate text against all stored names (reverse search)."""
    with ErrorHandler():
        store = get_store()
        text = smart_read(input_uri, mode="r")
        mentions = store.percolate(text, slop=slop)
        smart_write_models(output_uri, iter(mentions))


@cli.command("build")
def cli_build():
    with ErrorHandler():
        store = get_store()
        store.build()
