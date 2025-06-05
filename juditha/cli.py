from typing import Annotated, Optional

import typer
from anystore.cli import ErrorHandler
from rich import print

from juditha import __version__, io
from juditha.logging import configure_logging
from juditha.settings import Settings
from juditha.store import lookup

settings = Settings()

cli = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=settings.debug)


@cli.callback(invoke_without_command=True)
def cli_juditha(
    version: Annotated[Optional[bool], typer.Option(..., help="Show version")] = False,
):
    if version:
        print(__version__)
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


@cli.command("settings")
def cli_settings():
    with ErrorHandler():
        print(settings)
