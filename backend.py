from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import BayesIntentClassifier

app = FastAPI(title="BayesItent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = BayesIntentClassifier.from_json("atis_model_parameters.json")

class UserRequest(BaseModel):
    sentence: str


@app.post("/predict")
def predict_intent(request: UserRequest):
    results = model.predict_with_steps(request.sentence)
    return results


@app.get("/")
def read_root():
    return {"message": "BayesIntent Server is running!"}