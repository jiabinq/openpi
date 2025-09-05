# ruff: noqa
"""Offline tool to test which format works with your OpenPi server."""

import argparse
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

def test_format(client, format_name: str, format_obs: dict) -> bool:
    """Test if a specific format works with the server."""
    print(f"\nTesting {format_name} format...")
    try:
        response = client.infer(format_obs)
        if "actions" in response and response["actions"].shape == (16, 6):
            print(f"✅ {format_name} format works!")
            return True
    except Exception as e:
        print(f"❌ {format_name} format failed: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Test OpenPi format compatibility")
    parser.add_argument("--host", default="localhost", help="OpenPi server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPi server port")
    args = parser.parse_args()
    
    # Connect to server
    client = WebsocketClientPolicy(args.host, args.port)
    print(f"Connected to OpenPi server at {args.host}:{args.port}")
    
    # Prepare test data
    dummy_state = np.random.rand(6).astype(np.float32)
    dummy_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    
    # Test formats
    formats = {
        "phospho": {
            "observation/state": dummy_state,
            "observation/images.main.left": dummy_image,
            "observation/images.secondary_0": dummy_image,
            "prompt": "test task"
        },
        "libero": {
            "observation/state": dummy_state,
            "observation/image": dummy_image,
            "observation/wrist_image": dummy_image,
            "prompt": "test task"
        },
        "droid": {
            "observation/joint_position": dummy_state[:5],
            "observation/gripper_position": dummy_state[5:6],
            "observation/exterior_image_1_left": dummy_image,
            "observation/wrist_image_left": dummy_image,
            "prompt": "test task"
        }
    }
    
    # Test each format
    results = {}
    for name, obs in formats.items():
        results[name] = test_format(client, name, obs)
    
    # Summary
    print("\n" + "="*50)
    print("FORMAT COMPATIBILITY RESULTS:")
    print("="*50)
    for name, success in results.items():
        status = "✅ COMPATIBLE" if success else "❌ INCOMPATIBLE"
        print(f"{name:10s}: {status}")
    
    # Recommendation
    working_formats = [name for name, success in results.items() if success]
    if working_formats:
        print(f"\nRecommended format: {working_formats[0]}")
        print(f"Add to your config: openpi_format = '{working_formats[0]}'")
    else:
        print("\n⚠️ No formats worked! Check your server configuration.")

if __name__ == "__main__":
    main()