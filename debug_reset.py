import sys
import traceback
from fastapi.testclient import TestClient

try:
    from server.app import app
    client = TestClient(app)
    
    print("Sending POST to /reset...")
    response = client.post("/reset", params={"difficulty": "medium", "seed": 42})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"FAILED TO START OR RUN:")
    traceback.print_exc()
