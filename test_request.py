import requests
import json

reqUrl = "http://127.0.0.1:6000/predict_single"

headersList = {
 "Content-Type": "application/json" 
}

payload = json.dumps({
  "weak_authen":"temporary_work", "review":"This product is very bad!!"
})

response = requests.request("POST", reqUrl, data=payload,  headers=headersList)

print(response.text)