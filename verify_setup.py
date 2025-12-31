
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import api.main...")
    from api.main import app
    print("Successfully imported api.main")
except Exception as e:
    print(f"Failed to import api.main: {e}")
    sys.exit(1)

client = TestClient(app)

try:
    print("Testing /health endpoint...")
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("Health check passed!")
    else:
        print("Health check failed!")
        sys.exit(1)
except Exception as e:
    print(f"Error during request: {e}")
    sys.exit(1)
