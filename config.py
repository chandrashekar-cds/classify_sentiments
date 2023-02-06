import os

class GeneralConfig:
    API_KEY = "OPEN_AI_KEY"
    LOG_LEVEL = 'INFO'
    ## GPT3 allows 8000 for generating embeddings but later we are gonna use it for completion.create - which has a limit of about 3000 tokens - hence this limit
    embedding_models = {
                    "GPT3" : {"model" : "text-embedding-ada-002", "max_tokens":8191, "dimensions":1536},
                    "MPNet" : {"model" : "sentence-transformers/all-mpnet-base-v2", "max_tokens":384,"dimensions":784},
                    "DistilBERT" : {"model" : "sentence-transformers/multi-qa-distilbert-cos-v1", "max_tokens":512,"dimensions":784},
                    "DistilRoBERTa" : {"model" : "sentence-transformers/all-distilroberta-v1", "max_tokens":512,"dimensions":784},
                    "default" : {"model" : "sentence-transformers/multi-qa-mpnet-base-dot-v1", "max_tokens":512,"dimensions":784}
                    }

class ProdConfig(GeneralConfig):
    pass
   


class TestConfig(GeneralConfig):
    pass



class DevConfig(GeneralConfig):
    pass
   

def setConfig():
    env_name = os.getenv("APP_ENV", "dev")

    if env_name == "dev":
        config = DevConfig()
    elif env_name == "prod":
        config = ProdConfig()
    else:
        config = TestConfig()

    return config


Config = setConfig()
