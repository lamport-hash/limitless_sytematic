from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from visualiser_app.routes import data, chart
from visualiser_app.utils.enums import BUNDLE_DIR

app = FastAPI(
    title="DataFrame Visualiser",
    description="Visualise parquet dataframes from the bundle directory",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.bundle_dir = str(BUNDLE_DIR)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(data.router)
app.include_router(chart.router)


@app.get("/", response_class=HTMLResponse)
async def root():
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text()


@app.get("/health")
async def health():
    return {"status": "healthy", "bundle_dir": str(BUNDLE_DIR)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
