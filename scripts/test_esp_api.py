import requests
import json
import sys

# Replace with the IP address printed on your ESP32's Serial Monitor
ESP_IP = "192.168.1.100"  

def send_inference_request():
    url = f"http://{ESP_IP}/infer"
    
    # Generate a dummy input array (784 floats for MNIST/KWS)
    # In a real scenario, this would be your normalized image data
    print("Generating 784-element float array...")
    input_data = [0.5] * 784
    
    print(f"Sending POST request to {url}...")
    try:
        response = requests.post(url, json=input_data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        print("\n--- ESP32 Inference Results ---")
        print(f"Latency:      {result.get('latency_us', 0)} us")
        print(f"Free Memory:  {result.get('free_heap_kb', 0)} KB")
        print("Probabilities:")
        
        probs = result.get("probabilities", [])
        for i, p in enumerate(probs):
            print(f"  Class {i}: {p:.4f}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to ESP32: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ESP_IP = sys.argv[1]
    send_inference_request()
