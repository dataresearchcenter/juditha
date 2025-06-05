from anystore.io import smart_read
from fastapi import FastAPI
from fastapi.exceptions import HTTPException

from juditha import __version__, lookup
from juditha.settings import Settings
from juditha.store import Result

settings = Settings()


def get_description() -> str:
    return smart_read(settings.api.description_uri)


app = FastAPI(
    debug=settings.debug,
    title=settings.api.title,
    contact=settings.api.contact.model_dump(),
    description=get_description(),
    version=__version__,
    redoc_url="/",
)


@app.get("/{q}")
async def api_lookup(
    q: str,
    threshold: float | None = settings.fuzzy_threshold,
) -> Result | None:
    res = lookup(q, threshold=threshold)
    if res is None:
        raise HTTPException(status_code=404)
    return res


@app.head("/{q}")
async def api_head(q: str, threshold: float | None = settings.fuzzy_threshold) -> int:
    res = lookup(q, threshold=threshold)
    if res is None:
        raise HTTPException(404)
    return 200
