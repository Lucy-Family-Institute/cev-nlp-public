import multiprocessing
from datetime import datetime, timedelta
from typing import Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

import queue
import threading
from io import StringIO, BytesIO
import os
import zipfile
import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, Response, HTTPException, status, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
from os import listdir
from os.path import isfile, join
import pickle
from fastapi.params import Depends
from fastapi.responses import StreamingResponse

import sys

ROOT_PATH = os.path.dirname(
    '/'.join(os.path.abspath(__file__).split('/')[:-1]))
sys.path.insert(1, ROOT_PATH)

from root import DIR_INPUT, DIR_OUTPUT
from src.corpus.load_source_new import LoadSource
from src.corpus.build_source_new import BuildSource
from src.logic.notification import Notification


pairs = [
    ("person", "person"),
    ("organisation", "organisation"),
    ("person", "organisation"),
    ("person", "verb"),
    ("organisation", "verb"),
    ("organisation", "place"),
    ("chunk", "person"),
    ("chunk", "verb"),
    ("chunk", "organisation"),
    ("chunk", "adjective"),
    ("chunk", "place"),
]

types = ["verb", "person", "organisation", "adjective", "chunk", "noun-adj-pair", "place"]
# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "08a2efc05fdaeb9a0a7eb4e5fca34c0d3cde3cad23c18beb2339bbbc2986aaa0"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


list_users_db = {
    "liliana.pantoja": {
        "username": "liliana.pantoja",
        "full_name": "Liliana Pantoja",
        "hashed_password": "$2b$12$M4mARehv0IsMKF2RAphhAuMu6/q9vWGFY2zkoD6jZuefrUgfHiETi",
        "disabled": False,
    },
    "gabriel.moreno": {
        "username": "gabriel.moreno",
        "full_name": "Gabriel Moreno",
        "hashed_password": "$2b$12$q.8SxZeFl2BWpPO8CFIk7eNRt8L0SW8449B0OCBfojoTIBxKXkkgi",
        "disabled": False,
    },
    "leonardo.ibanez": {
        "username": "leonardo.ibanez",
        "full_name": "Leonardo Ibañez",
        "hashed_password": "$2b$12$duIL3FO.kK82qjwInU3ske1tVf7iqtNyICaAsDhVZMApkjXKYhkX.",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
current_process = {'step_0': '', 'step_1': '', 'step_2': '', 'step_3': '', 'step_4': ''}
queue_process = queue.Queue()

# Scheme for the Authorization header
token_auth_scheme = HTTPBearer()
app = FastAPI()
templates = Jinja2Templates(directory="templates")


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


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
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
    except JWTError:
        raise credentials_exception
    user = get_user(list_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class EntityNetwork:
    def __init__(self, process_name: str = None, process_file: str = None, emails: str = None):
        self.process_name = process_name
        self.process_file = process_file
        self.emails = emails
        self.out_path = f"{DIR_OUTPUT}/{process_name}"
        os.makedirs(self.out_path, exist_ok=True)

        out_files = os.listdir(self.out_path)
        if "load_source.p" not in out_files:
            self.ls = LoadSource(
                db_name=self.process_name,
                db_coll="document",
                path=self.process_file
            )
            self.ls.load_files_labels()
            with open(os.path.join(self.out_path, "load_source.p"), "wb") as f:
                pickle.dump(self.ls, f)
        else:
            with open(os.path.join(self.out_path, "load_source.p"), "rb") as f:
                self.ls = pickle.load(f)

        self.bs = BuildSource(self.ls, spacy_model_name="es_core_news_sm")
        if "build_source.p" not in out_files:
            self.bs.parse_documents()
            self.bs.build_big_table()
            self.bs.build_big_graph()
            self.bs.save_pickle(self.out_path)
        else:
            self.bs.load_pickle(self.out_path)

        # self.ls = LoadSource(
        #     db_name=self.process_name,
        #     db_coll="document",
        #     path=self.process_file
        # )
        # self.ls.load_files_labels()
        # self.bs = BuildSource(self.ls, spacy_model_name="es_core_news_sm")

    def build_source(self):
        # self.bs.parse_documents()
        # self.bs.build_big_table()
        # self.bs.build_big_graph()
        for type_ in types:
            self.bs.draw_frequent_words(
                type_,
                for_each_doc=False,
                path=os.path.join(self.out_path, f"{self.process_name}_freqs_{type_}.pdf"),
                topn=35,
                plotly=True,
                heading=f"{self.process_name}_freqs_{type_}"
            )
            print("Frequent words generated for", type_)

        for pair in pairs:

            cooc = list(self.bs.generate_cooccurrence_graph(*pair, for_each_doc=False))

            self.bs.draw_heatmaps(
                *pair,
                for_each_doc=False,
                path=os.path.join(
                    self.out_path, f"{self.process_name}_heatmap_{pair[0]}_vs_{pair[1]}.pdf"
                ),
                graphs=cooc,
                topn=35,
                plotly=True,
                heading=f"{self.process_name}_heatmap_{pair[0]}_vs_{pair[1]}"
            )

            print("Heatmaps generated for", pair)
            self.bs.draw_graphs(
                *pair,
                for_each_doc=False,
                path=os.path.join(self.out_path, f"{self.process_name}_graph_{pair[0]}_vs_{pair[1]}.html"),
                graphs=cooc,
                max_conns=50,
                width=100,
                height=100,
                percentage=True,
                plotly=False,
                heading=f"Network {self.process_name}: {pair[0]}_vs_{pair[1]}"
            )
            print("Graphs generated for", pair)

    def send_email(self):
        try:
            mails = self.emails.split()
            subject = f"End process {self.process_name}"
            content = f"The process with name {self.process_name} and file {self.process_file} has finished its execution."

            notification = Notification()
            notification.send(mails, subject, content)
        except Exception as e:
            print('Error: send_email')
            print(e)


def start_daemon_gatherer() -> None:
    """sets a never ending task for metering"""
    while True:
        item = queue_process.get()
        current_process.update(item)
        current_process['step_0'] = 'Start process in queue. ok.'
        g = EntityNetwork(process_file=item['process_file'], process_name=item['process_name'], emails=item['emails'])
        g.build_source()
        current_process['step_1'] = 'End process build_source. ok.'
        g.send_email()
        current_process['step_2'] = 'End process send email. ok.'
        queue_process.task_done()


@app.on_event("startup")
async def startup_event() -> None:
    """tasks to do at server startup"""
    t_queue = threading.Thread(target=start_daemon_gatherer)
    t_queue.start()


@app.get("/")
async def root():
    return {"message": "API's Social media reputation risk."}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(list_users_db, form_data.username, form_data.password)
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
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/pwd/{password}")
async def get_pwd(password: str):
    return {"message": pwd_context.hash(password)}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/get-queue/")
async def get_queue(current_user: User = Depends(get_current_active_user)):
    return {"message": get_process_status(), 'user': current_user.username}


def get_process_status():
    return f"The process queue has {queue_process.qsize()} jobs and the current process is {current_process}"


async def process(process_file: str, process_name: str, emails: str):
    """ Processing CSV and generate data"""
    queue_process.put({'process_file': process_file, 'process_name': process_name, 'emails': emails})


@app.post("/upload-to-process")
async def upload(background_tasks: BackgroundTasks, process_id: str = Form(...), emails: str = Form(...),
                 csv: UploadFile = File(...), current_user: User = Depends(get_current_active_user)):
    try:
        path_process = DIR_OUTPUT + process_id
        # if os.path.exists(path_process):
        #     raise HTTPException(status_code=404,
        #                         detail="This process id already exists, you need to rename the process.")
        #
        #file_name = "{}{}_{}.xlsx".format(DIR_INPUT, csv.filename.replace('.xlsx', ''),
        #                                    datetime.now().strftime('%Y-%m-%d_%H-%M'))
        #file_location = f"files/{csv.filename}"
        #with open(file_name, "wb+") as file_object:
        #    file_object.write(csv.file.read())

        # df = pd.read_excel(file_name)
        df = pd.read_excel(BytesIO(csv.file.read()))
        # print(df.describe)
        # print(df.columns)
        file_name = "{}{}_{}.pkl.gz".format(DIR_INPUT, csv.filename.replace('.xlsx', '').replace('.xls', ''),
                                            datetime.now().strftime('%Y-%m-%d_%H-%M'))
        df.to_pickle(file_name, compression="gzip")
        background_tasks.add_task(process, file_name, process_id, emails)
        return {"result": "CSV has been uploaded successfully", 'filename': csv.filename,
                'process_name': process_id, 'user': current_user.username,'status': True}
    except Exception as e:
        return {"result": str(e), 'filename': csv.filename,
                'process_name': process_id, 'status': False}


def zip_files(file_names, process_id):
    zip_subdir = DIR_OUTPUT + process_id + os.sep
    zip_filename = f"{process_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.zip"

    zip_io = BytesIO()
    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as temp_zip:
        for fpath in file_names:
            # Calculate path for file in zip
            fdir, fname = os.path.split(fpath)
            zip_path = os.path.join(zip_subdir, fname)
            # Add file, at correct path
            # print(fpath, zip_path)
            temp_zip.write(zip_path, fpath)
    return StreamingResponse(
        iter([zip_io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


@app.get("/get_process/{process_id}")
async def get_process(process_id: str, current_user: User = Depends(get_current_active_user)):
    path_process = DIR_OUTPUT + process_id
    if not os.path.exists(path_process):
        raise HTTPException(status_code=404, detail="Process_id not found")
    files_process = [f for f in listdir(path_process) if isfile(join(path_process, f))]
    return zip_files(files_process, process_id)


@app.get("/list_process/")
async def list_process(current_user: User = Depends(get_current_active_user)):
    files_process = [x[0].replace(DIR_OUTPUT, '') for x in os.walk(DIR_OUTPUT) if x[0].replace(DIR_OUTPUT, '') != '']
    return {'list_process': files_process, 'user': current_user.username}


@app.get("/graph_process/{process_id}")
async def get_graph_process(request: Request, process_id: str):
    path_process = DIR_OUTPUT + process_id
    if not os.path.exists(path_process):
        raise HTTPException(status_code=404, detail="Process_id not found")
    files_process = [f.replace('.html', '') for f in listdir(path_process) if isfile(join(path_process, f)) and join(path_process, f).find('.html') > 0]
    # filtered = filter(lambda file_single: file_single.find('.html') > 0, files_process)
    return templates.TemplateResponse('process.html',
                                      context={'request': request, 'result': {'name': process_id, 'list_files': files_process}})


@app.get("/graph_process/{process_id}/{graph_id}")
async def get_graph_process_id(request: Request, process_id: str, graph_id: str):
    path_process = f"{DIR_OUTPUT}{process_id}/{graph_id}.html"
    if not os.path.exists(path_process):
        raise HTTPException(status_code=404, detail="Process_id/graph_id not found")
    with open(path_process, "r", encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8002)

