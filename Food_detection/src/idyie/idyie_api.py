from datetime import timedelta, datetime, timezone
from typing import List
import requests
from torch.quantization import quantize_dynamic
from logging import StreamHandler
import logging
import os
from idyie.api_log import RouterLoggingMiddleware
from idyie.search_api_log import JsonFormatter
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import torch
import base64
from pydantic import BaseModel
from PIL import Image
import io
from typing import Annotated
import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client


app = FastAPI(title="Idyie", version="0.2.7", description="""API to expose computer vision model that detect food 
product on images and return a JSON with information about them. The main purpose is to help apps to automatize 
product cart creation.""")


apm = make_apm_client({
  'SERVICE_NAME': 'idyie_api',
  'SERVER_URL': 'http://192.168.1.80:8200'
})


# Schema for URL path to CDN.
class ImageUrl(BaseModel):
    url_path: str

# Schema for base64 encoded image.
class EncodedImage(BaseModel):
    encoded_image: str

# Schema for individual detection.
class Detection(BaseModel):
    name: str
    description: str
    category: str
    subcategory: str

# Schema for list of detections.
class DetectionsResponse(BaseModel):
    detections: List[Detection]


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


#os.environ["LOG_LEVEL"] = "DEBUG"
#os.environ["IDYIE_MODEL_PATH"] = "C:/Users/Utilisateur/Desktop/best.pt"


# Mapping for log level and env value
levels = {"CRITICAL": logging.CRITICAL,
          "ERROR": logging.ERROR,
          "WARNING": logging.WARNING,
          "INFO": logging.INFO,
          "DEBUG": logging.DEBUG}


def get_os_env(env_vars):
    """
     get_os_env(env_vars: list) ->

     Function to check if environment variable exist in the os.

     @param list env_vars: env_vars to check.
     """
    for var in env_vars:
        try:
            os.getenv(var)
        except KeyError:
            raise ValueError(f"Please set the environment variable {var}")


# Do not forget to set up theses 3 environment variables in the machine where the script will run
get_os_env(["LOG_LEVEL", "IDYIE_MODEL_PATH"])
level_env = os.getenv("LOG_LEVEL")
SECRET_KEY = os.getenv("FAST_SECRET_KEY")
ALGORITHM = os.getenv("FAST_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
FAST_USERNAME = os.getenv("FAST_USERNAME")
FAST_PASSWORD = os.getenv("FAST_PASSWORD")


fake_users_db = {
    FAST_USERNAME: {
        "username": FAST_USERNAME,
        "hashed_password": FAST_PASSWORD,
        "disabled": False
    }
}


torch.set_default_device("cpu")
torch.set_num_threads(8)

try:
    torch._logging.set_logs(all=levels[level_env])
except RuntimeError:
    pass

app.add_middleware(
    RouterLoggingMiddleware,
    logger=logging.getLogger(__name__)
)

app.add_middleware(ElasticAPM)


# Uncomment to quantize the model, it will reduce compute but lower the inferential performance.
# model = quantize_dynamic(model)

# Instantiate loggers
logger_search_api = logging.getLogger(__name__)
logger_search_api.setLevel(levels[level_env])

json_handler = StreamHandler()
json_formatter = JsonFormatter({"log.level": "levelname",
                                "message": "message",
                                "loggerName": "name",
                                "processName": "processName",
                                "processID": "process",
                                "threadName": "threadName",
                                "threadID": "thread",
                                "@timestamp": "asctime"})
json_handler.setFormatter(json_formatter)
logger_search_api.addHandler(json_handler)
model = YOLO(str(os.getenv("IDYIE_MODEL_PATH")))
logger_search_api.info(model.info())

product_info = {
    0: {"name": "pomme", "description": "Une pomme fraîche et juteuse.", "category": "fruit", "subcategory": "pomme"},
    1: {"name": "abricot", "description": "Un abricot mûr et sucré.", "category": "fruit",
        "subcategory": "fruit à noyau"},
    2: {"name": "banane", "description": "Une banane douce et mûre.", "category": "fruit",
        "subcategory": "fruit tropical"},
    3: {"name": "brocoli", "description": "Un brocoli vert et croquant.", "category": "légume",
        "subcategory": "crucifère"},
    4: {"name": "choux de Bruxelles", "description": "Petits choux verts savoureux.", "category": "légume",
        "subcategory": "crucifère"},
    5: {"name": "carotte", "description": "Une carotte croquante et sucrée.", "category": "légume",
        "subcategory": "racine"},
    6: {"name": "chou-fleur", "description": "Un chou-fleur blanc et tendre.", "category": "légume",
        "subcategory": "crucifère"},
    7: {"name": "piment rouge", "description": "Un piment rouge vif et épicé.", "category": "légume",
        "subcategory": "piment"},
    8: {"name": "pois chiche", "description": "Des pois chiches riches en protéines.", "category": "légume",
        "subcategory": "légumineuse"},
    9: {"name": "piment/chili", "description": "Un piment chili épicé.", "category": "légume", "subcategory": "piment"},
    10: {"name": "clémentine", "description": "Une clémentine juteuse et douce.", "category": "fruit",
         "subcategory": "agrume"},
    11: {"name": "concombre", "description": "Un concombre croquant et frais.", "category": "légume",
         "subcategory": "cucurbitacée"},
    12: {"name": "ail", "description": "De l'ail aromatique.", "category": "légume", "subcategory": "bulbe"},
    13: {"name": "raisin", "description": "Des raisins sucrés et juteux.", "category": "fruit",
         "subcategory": "fruit à pépins"},
    14: {"name": "citron", "description": "Un citron acidulé.", "category": "fruit", "subcategory": "agrume"},
    15: {"name": "champignon", "description": "Des champignons frais.", "category": "légume",
         "subcategory": "champignon"},
    16: {"name": "orange", "description": "Une orange douce et juteuse.", "category": "fruit", "subcategory": "agrume"},
    17: {"name": "pois", "description": "Des pois verts frais.", "category": "légume", "subcategory": "légumineuse"},
    18: {"name": "poire", "description": "Une poire juteuse et douce.", "category": "fruit",
         "subcategory": "fruit à pépins"},
    19: {"name": "kaki", "description": "Un kaki sucré et tendre.", "category": "fruit",
         "subcategory": "fruit exotique"},
    20: {"name": "cornichon", "description": "Des cornichons croquants.", "category": "légume",
         "subcategory": "cucurbitacée"},
    21: {"name": "ananas", "description": "Un ananas sucré et juteux.", "category": "fruit",
         "subcategory": "fruit tropical"},
    22: {"name": "pruneau", "description": "Un pruneau sec et sucré.", "category": "fruit",
         "subcategory": "fruit à noyau"},
    23: {"name": "citrouille", "description": "Une citrouille orange et sucrée.", "category": "légume",
         "subcategory": "cucurbitacée"},
    24: {"name": "framboise", "description": "Des framboises rouges et sucrées.", "category": "fruit",
         "subcategory": "baie"},
    25: {"name": "fraise", "description": "Des fraises rouges et juteuses.", "category": "fruit",
         "subcategory": "baie"},
    26: {"name": "tomate", "description": "Une tomate rouge et mûre.", "category": "légume",
         "subcategory": "solanacée"},
    27: {"name": "pastèque", "description": "Une pastèque sucrée et juteuse.", "category": "fruit",
         "subcategory": "cucurbitacée"},
    28: {"name": "viande rouge", "description": "De la viande rouge de haute qualité.", "category": "viande",
         "subcategory": "viande rouge"},
    29: {"name": "volaille", "description": "De la volaille tendre.", "category": "viande",
         "subcategory": "viande blanche"},
}


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/")
def access_docs():
    """
    ----------
    Description
    ----------
    Basic path to help you if you're lost.

    ----------
    Returns
    ----------
    Simple sentence to show to the user where to find the docs of the API.

     """
    logger_search_api.info("Accessing root path.")
    return "You can access to the documentation by reaching the path /docs or /redoc."


# Fonction pour traiter l'image avec le modèle PyTorch
def run_inference_pt(image: Image.Image, conf_threshold: float):
    """
    ----------
    Description
    ----------
    Function to run inference on the given image.

    ----------
    Params
    ----------
    @param conf_threshold: Threshold under which we don't return detected elements.
    @param image: Loaded image on which we will detect products.

    ----------
    Returns
    ----------
    JSON Response with the list of the elements detected in the image. {"detections": [...]}

    """
    results = model(image)
    detections = []
    detections_labels = []

    for i in range(len(results[0].boxes.cls)):
        label = int(results[0].boxes.cls[i])
        if (float(results[0].boxes.conf[i]) >= conf_threshold) & (label not in detections_labels):
            detections_labels.append(label)
            detections.append({
                "name": product_info[label]["name"],
                "description": product_info[label]["description"],
                "category": product_info[label]["category"],
                "subcategory": product_info[label]["subcategory"],
            })
    return JSONResponse(content={"detections": detections})


def get_image(url_path: str):
    """
    ----------
    Description
    ----------
    Endpoint to get JSON uploaded in CDN. (image base64 and threshold).

    ----------
    Params
    ----------
    @param str url_path: URL path which contain the JSON of the image base64 encoded and the threshold of the detection.

    ----------
    Returns
    ----------
    JSON Response with the base64 encoded image and the threshold. {"image_base64": ...., "threshold": ....}

    """
    image_json = requests.get(url_path).json()
    return image_json


@app.post("/detect/", response_model=DetectionsResponse, responses={
    200: {
        "description": "JSON with a list of JSON elements (unique food product detected) and their information.",
        "content": {
            "application/json": {
                "example": {
                    "detections": [
                        {
                            "name": "banane",
                            "description": "Une banane douce et mûre.",
                            "category": "fruit",
                            "subcategory": "fruit tropical"
                        },
                        {
                            "name": "fraise",
                            "description": "Des fraises rouges et juteuses.",
                            "category": "fruit",
                            "subcategory": "baie"
                        }
                    ]
                }
            }
        }
    }
})
async def detect_objects(image: ImageUrl,
                     current_user: Annotated[User, Depends(get_current_active_user)]):
    """
    ----------
    Description
    ----------
    Route to detect food products on images. You have to send an URL path in body (url_path).
    This url path is pointing to the CDN where is located the json with the base64 encoded image and the threshold of the detection. (image_base64 and threshold)

    ----------
    Params
    ----------
    @param image: Requête contenant l'url du json de l'image, qui comprend le threshold et l'image encodée en base64.

    ----------
    Returns
    ----------
    JSON Response with the list of the elements detected in the image. {"detections": [...]}

    """
    logger_search_api.info(f"Starting to process the request sent by {current_user.username}")
    try:
        image_json = get_image(image.url_path)
        # Décoder l'image à partir de base64
        image_data = base64.b64decode(image_json["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        logger_search_api.info("Image successfully decoded.")
    except Exception as e:
        logger_search_api.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data. Please ensure the image is base64 encoded.")

    # Passer l'image à la fonction d'inférence
    detections = run_inference_pt(image, image_json["threshold"])
    logger_search_api.info(f"Inference completed, results sent to {current_user.username}.")

    return detections


def run_srv():
    """
    Simple command to start to serve the API.
    """
    import uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    log_config["formatters"]["access"]["fmt"] = ("%(asctime)s %(levelprefix)s %(client_addr)s - "
                                                 "\"%(request_line)s\" %(status_code)s")
    logger_search_api.info("Server uvicorn run Idyie started.")
    uvicorn.run(app, host="0.0.0.0", port=8880, log_config=log_config, timeout_keep_alive=120)
    logger_search_api.info("Server uvicorn run Idyie execution finished.")


if __name__ == "__main__":
    run_srv()
