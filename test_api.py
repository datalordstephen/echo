import requests
import os

# Railway Deployment URL
API_URL = "https://echo-production-a131.up.railway.app/predict"
FILE_PATH = "example.wav"

def test_api():
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found.")
        return

    print(f"Testing API at {API_URL}...")
    print(f"Uploading {FILE_PATH}...")

    try:
        with open(FILE_PATH, 'rb') as f:
            files = {'file': (FILE_PATH, f, 'audio/wav')}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            print("\n✅ Success!")
            print("Response:")
            print(response.json())
        else:
            print(f"\n❌ Error {response.status_code}:")
            print(response.text)

    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    test_api()
