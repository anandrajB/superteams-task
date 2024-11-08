from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app import router


templates = Jinja2Templates(directory="templates", extensions=["jinja2.ext.do"])
origins = ["*"]


description = """
task in fastapi for image generation and fine tune. 🚀

## replicate-api

You will be able to:

* **generate images** 
* **fine tune**
"""


def create_app():
    app = FastAPI(
        title="Superteams task routes",
        description=description,
        summary="Replicate Image generation API's",
        version="1.0.0",
        contact={
            "name": "Anand Raj",
            "email": "anand98.ar@gmail.com",
        },
    )

    app.include_router(router.replicate_router, prefix="/v1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory="static"), name="static")

    return app


app = create_app()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("prompt.html", {"request": request})


@app.get("/generated_image/", response_class=HTMLResponse)
async def generated_image(request: Request):
    return templates.TemplateResponse("image.html", {"request": request})
