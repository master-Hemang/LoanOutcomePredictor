import requests

url = 'http://localhost:5000/predict'
data = {'user_id': 123}  

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")

try:
    response_data = response.json()
    print("Response JSON:")
    print(response_data)
except ValueError:
    print("Response is not valid JSON")
    print("Response Text:")
    print(response.text)
