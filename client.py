import requests
from pprint import pprint

url = "http://localhost:8000/predict"

# Input image path for the test
image_path = "images/beignets.png"

# Create the payload for the request
payload = {
    "image_path": image_path,
    "labels": ["a dog", "a cat", "a donut", "a beignet"],
}

# Send the request to the server
response = requests.post(url, json=payload)

# Print the predictions
pprint(response.json())
