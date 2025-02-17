import pymongo
import requests
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")

client = pymongo.MongoClient(MONGO_URI)
db = client.sample_mflix
collection = db.movies

embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

    return response.json()

query = "imaginary characters from outer space at war"

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "vector_index",
        }
    }
])

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')
