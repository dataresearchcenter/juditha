from typing import Annotated, Optional

import typer
from anystore.cli import ErrorHandler
from anystore.logging import configure_logging
from rich import print

from juditha import __version__, io
from juditha.settings import Settings
from juditha.store import get_store, lookup, validate_name

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


@cli.command("validate")
def cli_validate(
    value: str, tag: Annotated[str, typer.Option("-t", help="PER, ORG, LOC")]
):
    with ErrorHandler():
        result = validate_name(value, tag)
        print(result)


@cli.command("build")
def cli_build():
    with ErrorHandler():
        store = get_store()
        store.build()
