# Phospho Format Validation Strategy

## Problem: Untested Format Conversion

Phospho's SO-100 format conversion is **untested by you** but we need to trust it for OpenPi server communication. We have:

- ✅ **Tested**: LeRobot record functionality
- ✅ **Tested**: OpenPi DROID/LIBERO server formats  
- ❌ **Untested**: Phospho's specific SO-100 format conversion

## Validation Approach: Cross-Reference Against Known Working Formats

### **Reference Formats Analysis**

| Source | State Format | Image Format | Status |
|--------|-------------|-------------|---------|
| **Phospho SO-100** | `"observation/state"` (6D) | `"observation/images.main.left"` | ❌ **Untested** |
| **OpenPi DROID** | `"observation/joint_position"` (7D) + `"observation/gripper_position"` (1D) | `"observation/exterior_image_1_left"` | ✅ **Tested** |
| **OpenPi LIBERO** | `"observation/state"` (8D) | `"observation/image"` | ✅ **Tested** |
| **LeRobot Aug** | `"state"` (6D) | `"base"`, `"wrist"` | ✅ **Tested** |

### **Key Findings**

1. **State Format**: Phospho uses **same key as LIBERO** (`"observation/state"`) ✅
2. **Image Format**: Phospho uses **unique keys** not found in tested formats ❌
3. **Dimensions**: Phospho's 6DOF matches LeRobot Aug ✅

## Validation Strategy

### **Phase 1: Format Key Validation**

**Test Phospho's keys against OpenPi server using LIBERO format as reference:**

```python
# Tested LIBERO format (known to work)
libero_obs = {
    "observation/state": np.random.rand(8),                                    # ✅ Key works
    "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),   # ✅ Key works  
    "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # ✅ Key works
    "prompt": "do something",
}

# Phospho format (unknown)
phospho_obs = {
    "observation/state": np.random.rand(6),                                    # ✅ Same key as LIBERO
    "observation/images.main.left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),    # ❓ Different key
    "observation/images.secondary_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # ❓ Different key  
    "prompt": "do something",
}
```

**Risk**: Phospho's image keys might not be recognized by OpenPi server.

### **Phase 2: Progressive Format Testing**

**2.1 Test State Format (Low Risk)**
```python
def test_state_format():
    """Test if Phospho's state format works - should work since LIBERO uses same key."""
    test_obs = {
        "observation/state": np.random.rand(6),  # Phospho format
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # LIBERO image key
        "prompt": "test state format"
    }
    
    try:
        response = policy_client.infer(test_obs)
        print("✅ Phospho state format works")
        return True
    except Exception as e:
        print(f"❌ Phospho state format failed: {e}")
        return False
```

**2.2 Test Image Keys (High Risk)**
```python  
def test_image_keys():
    """Test if Phospho's image keys work."""
    
    # Test primary image key
    test_obs_main = {
        "observation/state": np.random.rand(6),
        "observation/images.main.left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "test main image"
    }
    
    try:
        response = policy_client.infer(test_obs_main)
        print("✅ Phospho main image key works")
        main_works = True
    except Exception as e:
        print(f"❌ Phospho main image key failed: {e}")
        main_works = False
    
    # Test secondary image key  
    test_obs_secondary = {
        "observation/state": np.random.rand(6),
        "observation/images.secondary_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "test secondary image"
    }
    
    try:
        response = policy_client.infer(test_obs_secondary)
        print("✅ Phospho secondary image key works")  
        secondary_works = True
    except Exception as e:
        print(f"❌ Phospho secondary image key failed: {e}")
        secondary_works = False
        
    return main_works, secondary_works
```

### **Phase 3: Fallback Format Mapping**

**If Phospho's image keys fail, map to known working keys:**

```python
def get_validated_format_converter():
    """Return format converter with validated keys."""
    
    # Test Phospho's format
    main_works, secondary_works = test_image_keys()
    
    if main_works and secondary_works:
        print("✅ Using Phospho's original format")
        return PhosphoFormatConverter()
    else:
        print("⚠️ Using fallback format mapping")
        return FallbackFormatConverter(main_works, secondary_works)

class FallbackFormatConverter:
    def __init__(self, main_works: bool, secondary_works: bool):
        self.main_works = main_works
        self.secondary_works = secondary_works
        
    def lerobot_to_openpi(self, raw_obs: dict, task: str) -> dict:
        """Convert with validated key mapping."""
        
        # State: Use Phospho's key (validated to work like LIBERO)
        state = np.array([
            raw_obs["shoulder_pan.pos"],
            raw_obs["shoulder_lift.pos"], 
            raw_obs["elbow_flex.pos"],
            raw_obs["wrist_flex.pos"],
            raw_obs["wrist_roll.pos"],
            raw_obs["gripper.pos"],
        ])
        
        openpi_obs = {
            "observation/state": state,  # ✅ Validated to work
            "prompt": task
        }
        
        # Image mapping with fallbacks
        cameras = ["front", "wrist", "base"]  # Common LeRobot camera names
        
        for i, camera_name in enumerate(cameras):
            if camera_name in raw_obs and isinstance(raw_obs[camera_name], np.ndarray):
                image = cv2.resize(raw_obs[camera_name], (224, 224))
                
                if i == 0:  # Primary camera
                    if self.main_works:
                        openpi_obs["observation/images.main.left"] = image
                    else:
                        # Fallback to LIBERO format
                        openpi_obs["observation/image"] = image
                        print("⚠️ Using LIBERO image key fallback")
                        
                else:  # Secondary cameras
                    if self.secondary_works:
                        openpi_obs[f"observation/images.secondary_{i-1}"] = image
                    else:
                        # Fallback to DROID wrist format
                        openpi_obs["observation/wrist_image_left"] = image  
                        print("⚠️ Using DROID wrist key fallback")
                        break  # Only one wrist camera in DROID format
                        
        return openpi_obs
```

### **Phase 4: Live Testing Protocol**

**Test with real policy server before full evaluation:**

```python
def validate_phospho_format_live(policy_host: str, policy_port: int):
    """Validate Phospho format against live OpenPi server."""
    
    print("🔬 Validating Phospho format conversion...")
    
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy(policy_host, policy_port)
    
    # 1. Test server connectivity with known format
    print("1️⃣ Testing server connectivity with LIBERO format...")
    libero_obs = _random_observation_libero()  # From simple_client/main.py
    try:
        response = client.infer(libero_obs)
        print("✅ Server connectivity confirmed")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False
    
    # 2. Test Phospho's state format
    print("2️⃣ Testing Phospho state format...")
    state_works = test_state_format(client)
    
    # 3. Test Phospho's image keys  
    print("3️⃣ Testing Phospho image keys...")
    main_works, secondary_works = test_image_keys(client)
    
    # 4. Test complete Phospho format
    print("4️⃣ Testing complete Phospho format...")
    complete_phospho_obs = {
        "observation/state": np.random.rand(6),
        "observation/images.main.left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "full phospho format test"
    }
    
    try:
        response = client.infer(complete_phospho_obs)
        print("✅ Complete Phospho format works!")
        return True
    except Exception as e:
        print(f"❌ Complete Phospho format failed: {e}")
        print("ℹ️ Will use fallback mapping")
        return False

# Usage before main evaluation
if __name__ == "__main__":
    config = SO100EvalConfig()
    
    # Validate format before starting evaluation  
    format_works = validate_phospho_format_live(config.policy_host, config.policy_port)
    
    if format_works:
        print("🎉 Using Phospho's format directly")
        converter = PhosphoFormatConverter()
    else:
        print("🔄 Using validated fallback format")  
        converter = get_validated_format_converter()
        
    # Proceed with evaluation using validated converter
    evaluator = SO100Evaluator(config, converter)
    evaluator.run_episodes()
```

## Validation Results Matrix

| Test | Expected Result | Action if Failed |
|------|----------------|------------------|
| **Server Connectivity** | ✅ | Abort - fix server setup |
| **State Format** | ✅ (same as LIBERO) | Abort - investigate server |
| **Main Image Key** | ❓ | Fallback to `"observation/image"` |
| **Secondary Image Key** | ❓ | Fallback to `"observation/wrist_image_left"` |
| **Complete Format** | ❓ | Use FallbackFormatConverter |

## Implementation Priority

1. **High Priority**: Implement format validation before any robot testing
2. **Medium Priority**: Create fallback converters for each failure case  
3. **Low Priority**: Optimize format performance after validation

## Expected Issues & Solutions

### **Issue 1: Image Key Not Recognized**
```
Error: "observation/images.main.left" not found in expected keys
```
**Solution**: Map to `"observation/image"` (LIBERO) or `"observation/exterior_image_1_left"` (DROID)

### **Issue 2: Secondary Image Ignored**
```  
Server ignores "observation/images.secondary_0"
```
**Solution**: Map to `"observation/wrist_image_left"` (DROID) or `"observation/wrist_image"` (LIBERO)

### **Issue 3: State Dimension Mismatch**
```
Expected state dimension 8, got 6
```
**Solution**: Pad state like Phospho's transform does: `pad_to_dim(state, 8)`

## Summary

**Validation strategy protects against Phospho format failures by:**

1. ✅ **Progressive testing** - Test components before full format
2. ✅ **Fallback mapping** - Use known working keys if Phospho's keys fail  
3. ✅ **Live validation** - Test against real server before evaluation
4. ✅ **Graceful degradation** - Continue evaluation even with format fallbacks

**This ensures SO-100 evaluation works regardless of Phospho format correctness.**