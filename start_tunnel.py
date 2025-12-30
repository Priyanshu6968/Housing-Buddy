from pyngrok import ngrok
import time

# Open a HTTP tunnel on port 8000
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down tunnel...")
    ngrok.disconnect(public_url)
