import requests

url = "http://localhost:6006/v1/rerank"
headers = {"Authorization": "Bearer sk-xxxx"}

data = {
    "query": "What is the capital of China?",
    "documents": [
        "The capital of China is Beijing.",
        "China is a country in East Asia.",
        "Beijing is a large city with many historical sites."
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())