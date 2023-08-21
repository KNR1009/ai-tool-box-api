from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_data_from_jsonplaceholder():
    response = requests.get("https://jsonplaceholder.typicode.com/todos")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data from jsonplaceholder"}
