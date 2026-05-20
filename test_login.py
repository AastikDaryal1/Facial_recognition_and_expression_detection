import urllib.request
import urllib.parse
import json

url = 'http://localhost:8000/auth/login'
data = {'email': 'shubhsonakiya86@gmail.com', 'password': 'Shubh@86'}
data_json = json.dumps(data).encode('utf-8')
req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'}, method='POST')

try:
    with urllib.request.urlopen(req) as response:
        result = response.read().decode('utf-8')
        print(f"Success: {result}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
