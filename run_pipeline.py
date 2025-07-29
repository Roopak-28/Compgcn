import requests
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
from pydantic import BaseModel

class PipelineConfig(BaseModel):
    pipeline_name: str = "piercing_jun23a"
    pipeline_description: str = "ML pipeline for piercing"
    pipeline_json: Dict[str, Any] = {}
    


def get_access_token():
    url = "https://ig.aidtaas.com/mobius-iam-service/v1.0/login"

    payload = json.dumps({
      "userName": "aidtaas@gaiansolutions.com",
      "password": "Gaian@123",
      "productId": "c2255be4-ddf6-449e-a1e0-b4f7f9a2b636",
      "requestType": "TENANT"
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()
    # print(response)
    # print(response['accessToken'])
    access_token = response['accessToken']

    return access_token


def create_pipeline(config):
    url = "https://ig.aidtaas.com/bob-service-test/v1.0/pipeline"

    payload = json.dumps({
      "name": config["pipeline_name"],
      "description": config['pipeline_description'],
      "jsonInput": [
        config['pipeline_json']
      ],
      "pipelineType": "ML"
    })
    access_token = get_access_token()
    headers = {
      'accept': 'application/json',
      'Authorization': f"Bearer {config['access_token']}",
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.text)
    print(response)
    response = response.json()
    print(response)
    return response['pipelineId']


def trigger_pipeline(config):
    url = f"https://ig.aidtaas.com/bob-service-test/v1.0/pipeline/trigger/ml?pipelineId={config['pipeline_id']}"

    payload = json.dumps({
    "pipelineType": "ML",
    "containerResources": {},
    "experimentId": config['experiment_id'],
    "enableCaching": True,
    "parameters": {},
    "version": 1
    })
    headers = {
    'accept': 'application/json',
    'Authorization': f"Bearer {config['access_token']}",
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()
    print(response)
    # print(response.text)
    return response['runId']

app = FastAPI()

@app.post("/run-pipeline")
def run_pipeline(config: PipelineConfig):
    config = config.dict()
    config['experiment_id'] = "1ec485ba-e6ed-4fa7-aa41-d4b8d5841a08"
    config['access_token'] = get_access_token()
    pipeline_id = create_pipeline(config)
    config["pipeline_id"] = pipeline_id
    run_id = trigger_pipeline(config)
    result = {
        "pipeline_id" : pipeline_id,
        "run_id" : run_id
    }
    return result 

# async def run_pipeline_endpoint(config: PipelineConfig):
#     try:
#         # Convert the validated Pydantic model to a dict for downstream functions
#         result = run_pipeline(config.dict())
#         return JSONResponse(content=result)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("run_pipeline:app", host="127.0.0.1", port=8000, reload=True)




