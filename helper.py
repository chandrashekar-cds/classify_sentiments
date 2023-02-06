import re
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
import openai
from openai.embeddings_utils import get_embedding
from config import Config
openai.api_key = Config.API_KEY
import logging
logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)
logging.getLogger("openai").setLevel(logging.WARNING)
import tiktoken
#import backoff
import pandas as pd
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embeddings(text,engine):
    try:
        return get_embedding(text,engine)
    except openai.error.RateLimitError as e:
        raise Exception("Rate limit error: {}".format(str(e)))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#num_tokens_from_string("tiktoken is great!", "cl100k_base")

def embed_gen_model(model_name):
    engine = Config.embedding_models[model_name]['model']
    MAX_TOKENS = Config.embedding_models[model_name]['max_tokens']
    dimensions = Config.embedding_models[model_name]['dimensions']
    if model_name == "GPT3":
        model = engine
    else:
        model = SentenceTransformer(engine)
    return model, MAX_TOKENS, dimensions

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
# def embeddings(text,engine):
#     return get_embedding(text,engine)



def get_embeddings(text,model):
    # Generate the embedding
    embedding = model.encode([text])
    logger.info(embedding.shape)
    return np.array(embedding[0])

def is_value_present(dictionaries, key, value):
    for dictionary in dictionaries:
        if dictionary.get(key) == value:
            return True
    return False

def convertDF2JSON(df: "pd.DataFrame"):
    """
    
    Helper function to convert dataframe to JSON
    
    """
    # float64_cols = list(df.select_dtypes(include='float64'))
    # df[float64_cols] = df[float64_cols].astype('float32')
    df.fillna('', inplace = True)
    return df.to_dict('records')

class Item(BaseModel):
    review: str
    about: str = None
  
    class Config:
        example = {
            "review": "This movie is something what I wanted to watch for many months.. after watching it I am neither happy nor sad.. a watchable movie.",
            "about": "A sample review to assess the sentiment"
        }

class auth(BaseModel):
    weak_authen:str
    class Config:
        example = {
            "weak_authen":"temporary_work"
        }