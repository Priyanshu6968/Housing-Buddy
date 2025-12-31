from pyngrok import ngrok
import time
import os

# Determine port to expose (match the FastAPI default in api/main.py)
port = int(os.environ.get("PORT") or os.environ.get("APP_PORT") or 9000)

# Set ngrok auth token from environment if provided
auth_token = os.environ.get("NGROK_AUTH_TOKEN") or os.environ.get("NGROK_AUTHTOKEN")
if auth_token:
    try:
        ngrok.set_auth_token(auth_token)
    except Exception:
        print("Warning: failed to set ngrok auth token; continuing without token")
else:
    print("No NGROK_AUTH_TOKEN found in environment; public tunnels may be limited")

# Open an HTTP tunnel on the selected port
public_url = ngrok.connect(port, "http").public_url
print(f"Public URL: {public_url}")

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down tunnel...")
    try:
        ngrok.disconnect(public_url)
    except Exception:
        pass
