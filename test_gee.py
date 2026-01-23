import requests
import json

def test_gee(lat=12.9716, lon=77.5946):
    url = f"http://localhost:8001/get_field_health?lat={lat}&lon={lon}"
    print(f"Testing GEE endpoint with: lat={lat}, lon={lon}")
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"Error returned: {data['error']}")
            else:
                print("Success!")
                # print(json.dumps(data, indent=2))
        else:
            print(f"Failed: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_gee() # Bangalore
    test_gee(lat=18.5204, lon=73.8567) # Pune
    test_gee(lat=0, lon=0) # Middle of ocean (likely to fail)
