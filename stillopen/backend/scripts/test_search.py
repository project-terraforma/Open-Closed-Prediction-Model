import httpx
import time

def run_tests():
    # To run this script, uvicorn must be running locally on http://localhost:8000
    base_url = "http://localhost:8000"
    
    print("Testing /health ... ", end="")
    try:
        res = httpx.get(f"{base_url}/health")
        if res.status_code == 200:
            print("OK")
        else:
            print("Failed")
    except Exception as e:
        print("Failed to connect!")
        return
        
    print("\n--- Testing pg_trgm Fuzzy Search (McDonalds) ---")
    start = time.time()
    res = httpx.get(f"{base_url}/search?q=McDonalds&limit=3")
    print(f"Latency: {(time.time() - start) * 1000:.2f}ms")
    if res.status_code == 200:
        data = res.json()
        print(f"Results Count: {len(data)}")
        for idx, place in enumerate(data):
            print(f"{idx+1}. {place['name']} - conf: {place['confidence']:.2f}")
    else:
        print(f"Error {res.status_code}: {res.text}")

    print("\n--- Testing pg_trgm Spatial Filtering (Coffee in BBOX) ---")
    # A pseudo-filter for California bounds
    start = time.time()
    res = httpx.get(f"{base_url}/search?q=Coffee&min_lat=32.0&max_lat=42.0&min_lon=-124.0&max_lon=-114.0&limit=3")
    print(f"Latency: {(time.time() - start) * 1000:.2f}ms")
    if res.status_code == 200:
        data = res.json()
        print(f"Results Count: {len(data)}")
        for idx, place in enumerate(data):
            print(f"{idx+1}. {place['name']} (Lat: {place['lat']} Lon: {place['lon']})")
    else:
        print(f"Error {res.status_code}: {res.text}")

if __name__ == "__main__":
    run_tests()
