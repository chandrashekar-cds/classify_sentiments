# Code from : https://github.com/tiangolo/fastapi/issues/364

## Main
from fastapi import FastAPI
from typing import Any, Dict, List, Union
from config import Config
import uvicorn, os
## Security Basic Auth
import secrets
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import UploadFile, File
## Add Doc Security
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import pickle
import concurrent.futures
from helper import *
from tqdm import tqdm

model = {}
def load_model(MODEL:str):
    global model
    with open(f"{MODEL}.pickle", "rb") as file:
        model = pickle.load(file)

## Initial App
app = FastAPI(
    title="Sentiment Classifier",
    description="Classifies a given review as positive or negative",
    version="0.88.0",
    docs_url=None,
    redoc_url=None,
    openapi_url = None,
)

## Initial Basic Auth
security = HTTPBasic()
## For document
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "user")
    correct_password = secrets.compare_digest(credentials.password, "password")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

## For actual API => to aviod Authentication header from GCP's bearer token
def weak_authentication(cred : auth):
    credentials = cred.__dict__
    weak_key = credentials.get('weak_authen','')
    if weak_key != "temporary_work":
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Basic"},
            )
    return 'authen_ok'

## Create Auth Doc
@app.get("/openapi.json", include_in_schema=False)
async def openapi(username: str = Depends(get_current_username)):
    return get_openapi(title=app.title, version=app.version, routes=app.routes)

@app.route('/')
@app.get("/docs", include_in_schema=False)
async def get_swagger_documentation(username: str = Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation(username: str = Depends(get_current_username)):
    return get_redoc_html(openapi_url="/openapi.json", title="docs")
    
## API Function, use POST because GCP's authentication does not allow body in GET
@app.post("/myapi")
async def myapi(username: str = Depends(weak_authentication)):
    return {'status' : username, 'hello' : 'world'}

@app.post("/predict_single")
#async def predict_single(username: str = Depends(weak_authentication),request: Union[List,Dict,Any]=None):
async def predict_single(request: Item):
    #print(f"request contents:{request}")
    request_body = request.__dict__
    engine, MAX_TOKENS,dimensions = embed_gen_model("GPT3")
    review = str(request_body['review'])
    if review:
        query = [embeddings(review,engine)]
        query = np.array(query)
        query_df = pd.DataFrame(query, columns=[f"embed_{i}" for i in range(1536)])
        #print(query_df.shape)
        prediction = model.predict(query_df)
        sentiment = 'positive' if prediction == 1 else 'negative'
        return {'review':request_body['review'],'sentiment':sentiment}
    else:
        return {"Empty strings can't have any sentiments!"}

@app.post("/predict_bulk")
#async def predict_single(username: str = Depends(weak_authentication),request: Union[List,Dict,Any]=None):
async def predict_bulk(
        file: UploadFile = File(description="A csv file containing reviews under column heading 'review'")
        ):
    engine, MAX_TOKENS,dimensions = embed_gen_model("GPT3")
    df = pd.read_csv(file.file)
    if 'embeddings' not in df.columns:
        df['embeddings'] = pd.Series(dtype=object)
    no_of_rows = df['embeddings'].isna().sum()
    print(f"Total embeddings to be generated = {no_of_rows}")
    #print(f"Total embeddings to be generated = {no_of_rows}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with tqdm(total=no_of_rows) as pbar:
            future_to_index = {executor.submit(embeddings, row['review'],engine): i for i, row in df.iterrows() if row['review']}
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    embedding = future.result()
                    # console_logger.debug(f"embeddings of {i} = {embedding}")
                    # console_logger.debug(f"shape of df = {df.shape}")#, shape of embeddings = {embeddings.shape}")
                    df.at[i, "embeddings"] = embedding
                    pbar.update()
                    if (i+1) % 100 == 0:
                        df.to_csv(f"partial_embeddings.csv", index=False)
                        #print(f"{i+1} embeddings generated and saved to partial_embeddings.csv.")
                        print(f"{i+1} embeddings generated and saved to partial_embeddings.csv.")
                except Exception as exc:
                    df.to_csv(f"partial_embeddings.csv", index=False)
                    print(f"An error occurred: {exc}. Saving partial embeddings to partial_embeddings.csv and exiting.")
    
    df1 = pd.DataFrame()
    df1 = df.dropna(subset=['embeddings'])#.sample(8000)
    # df1 = df1.copy()
    # df1['embeddings'] = df1.embeddings.apply(eval).apply(np.array)
    X = pd.concat([pd.DataFrame(df1['embeddings'].to_list(), columns=[f"embed_{i}" for i in range(1536)])], axis=1)
    print(X.shape)
    prediction = model.predict(X)
    results = pd.DataFrame()
    results['review'] = df['review'] 
    results['sentiment'] = ["positive" if value == 1 else "negative" for value in prediction]
    # return {'review':request['review'],'sentiment':sentiment}
    return  convertDF2JSON(results)


## RUN!!!
if __name__ == "__main__":
    MODEL = os.environ.get("MODEL",'ensemble')
    print(f"* Loading {MODEL} model and starting Fastapi server...")
    load_model(MODEL)
    uvicorn.run(app ,port=int(os.environ.get("port", 6000)), host="0.0.0.0")