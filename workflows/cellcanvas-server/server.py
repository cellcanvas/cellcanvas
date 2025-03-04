# /// script
# title = "CellCanvas-enabled Copick Server"
# description = "Run a Copick server with CellCanvas VAE feature extraction and reconstruction capabilities"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["server", "deep learning", "cell imaging", "microscopy"]
# repository = "https://github.com/kephale/cellcanvas"
# documentation = "https://github.com/kephale/cellcanvas#readme"
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.9",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "torch",
#     "copick",
#     "copick-server @ git+https://github.com/kephale/copick-server",
#     "click",
#     "uvicorn",
#     "starlette",
#     "zarr<3",
# ]
# ///

import os
import sys

import os
import json
import click
import uvicorn
import copick
from pathlib import Path
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.routing import Mount, Route

# Import original Copick server components
from copick_server.server import CopickRoute, create_copick_app

def create_cellcanvas_copick_app(root, cors_origins=None):
    """Create a Starlette app that includes both Copick and CellCanvas routes."""
    # Create the original Copick route handler
    copick_handler = CopickRoute(root)
    
    routes = [
        # Debug route
        Route("/debug", endpoint=lambda request: Response(
            json.dumps({
                "runs": list(root.get_run_list()) if hasattr(root, "get_run_list") else [],
                "root_type": str(type(root).__name__),
                "root_dir": str(dir(root))
            }),
            media_type="application/json"
        )),

        # Original Copick catchall route - MOVE TO END
        Route("/{path:path}", endpoint=copick_handler.handle_request, methods=["GET", "HEAD", "PUT"]),
    ]
    
    # Create Starlette app
    app = Starlette(routes=routes)
    
    # Add CORS middleware if needed
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    return app

def serve_cellcanvas_copick(config_path, allowed_origins=None, **kwargs):
    """Start an HTTP server serving a Copick project with CellCanvas extensions."""
    root = copick.from_file(config_path)
    app = create_cellcanvas_copick_app(root, allowed_origins)
    uvicorn.run(app, **kwargs)

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--cors",
    type=str,
    default=None,
    help="Origin to allow CORS. Use wildcard '*' to allow all.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Bind socket to this host.",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Bind socket to this port.",
    show_default=True,
)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def main(config, cors, host, port, reload):
    """Serve a Copick project with CellCanvas features over HTTP."""
    print(f"Starting CellCanvas-enabled Copick server with config: {config}")

    serve_cellcanvas_copick(
        config,
        allowed_origins=[cors] if cors else None,
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    main()