from enum import StrEnum
from typing import Annotated, Optional

import typer
from anystore.cli import ErrorHandler
from anystore.io import smart_open, smart_stream, smart_write_models
from anystore.logging import configure_logging
from rich import print

from juditha import __version__, io, predict
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


DEFAULT_FT_MODEL = str(settings.make_path("schema_classifier.bin"))


@cli.command("fasttext-sample")
def cli_fasttext_sample(
    output_uri: Annotated[
        str, typer.Option("-o", help="Output uri, default stdout")
    ] = "-",
    limit: Annotated[int, typer.Option("-l", help="Sample size")] = 100_000,
):
    """Iterate through names.db and get sampled fasttext classifier data"""
    with ErrorHandler():
        store = get_store()
        samples = predict.get_sample(store.aggregator, limit=limit)
        lines = (" ".join(s) + "\n" for s in samples)
        with smart_open(output_uri, "w") as out:
            out.writelines(lines)


@cli.command("fasttext-train")
def cli_fasttext_train(
    output_uri: Annotated[
        str, typer.Option("-o", help="Model output uri")
    ] = DEFAULT_FT_MODEL,
    limit: Annotated[int, typer.Option("-l", help="Sample size")] = 100_000,
    epoch: Annotated[int, typer.Option("--epoch", help="Number of epochs")] = 25,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 0.1,
    word_ngrams: Annotated[int, typer.Option("--word-ngrams", help="Word n-grams")] = 2,
    dim: Annotated[int, typer.Option("--dim", help="Embedding dimension")] = 100,
    ws: Annotated[int, typer.Option("--ws", help="Window size")] = 5,
    min_count: Annotated[
        int, typer.Option("--min-count", help="Minimum word count")
    ] = 1,
    verbose: Annotated[int, typer.Option("--verbose", help="Verbosity level")] = 2,
):
    """Generate a trained fasttext schema classifier model"""
    with ErrorHandler():
        store = get_store()
        predict.train_model(
            store.aggregator,
            model_path=output_uri,
            limit=limit,
            epoch=epoch,
            lr=lr,
            wordNgrams=word_ngrams,
            dim=dim,
            ws=ws,
            minCount=min_count,
            verbose=verbose,
        )


class Formats(StrEnum):
    json = "json"
    csv = "csv"


@cli.command("fasttext-predict")
def cli_fasttext_predict(
    input_uri: Annotated[
        str, typer.Option("-i", help="Input uri, default stdin")
    ] = "-",
    output_uri: Annotated[
        str, typer.Option("-o", help="Output uri, default stdout")
    ] = "-",
    output_format: Formats | None = Formats.json,
    model_path: Annotated[
        str, typer.Option("-m", help="Model path")
    ] = DEFAULT_FT_MODEL,
):
    """Predict schema for given text using FastText model"""
    with ErrorHandler():

        def _get_results():
            for text in smart_stream(input_uri, mode="r"):
                yield from predict.predict_schema(text.strip(), model_path=model_path)

        smart_write_models(output_uri, _get_results(), output_format=output_format)


@cli.command("build")
def cli_build():
    with ErrorHandler():
        store = get_store()
        store.build()
