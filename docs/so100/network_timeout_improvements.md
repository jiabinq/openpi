# Network Timeout Improvements for SO-100 Implementation

## Issue Description

Network timeout during critical robot movement can leave the robot in unsafe positions. This affects both our SO-100 implementation and DROID - it's a common issue with remote inference systems.

## Problem Location

```python
# Lines 388-399 in SO-100 implementation plan
if self.actions_from_chunk_completed == 0 or self.actions_from_chunk_completed >= self.config.open_loop_horizon:
    # CRITICAL NETWORK CALL - can timeout here
    with prevent_keyboard_interrupt():
        response = self.policy_client.infer(openpi_obs)  # ← Network timeout risk
        self.pred_action_chunk = response["actions"]
```

## Risk Scenarios

1. **Pick and Place**: Robot holding object when network fails → object dropped
2. **Approach Movement**: Robot approaching target when network fails → potential collision
3. **Retraction**: Robot retracting when network fails → blocks workspace

## Current Status

- ✅ **Same issue exists in DROID** - this is a known limitation of remote inference
- ✅ **LeRobot has same issue** when using remote policies
- ✅ **Start simple approach is correct** - get basic functionality working first
- ✅ **Address after first success** - optimization for production use

## Future Improvement Options (After First Success)

### **Option 1: Timeout with Safe Action**
```python
try:
    response = asyncio.wait_for(
        policy_client.infer(openpi_obs), 
        timeout=2.0  # 2 second timeout
    )
    self.pred_action_chunk = response["actions"]
except asyncio.TimeoutError:
    # Use safe stop action
    print("Network timeout - executing safe stop")
    self.pred_action_chunk = np.zeros((self.config.action_horizon, 6))  # Safe stop
    self.actions_from_chunk_completed = 0
```

### **Option 2: Local Action Buffering**
```python
class ActionBuffer:
    def __init__(self, buffer_size=3):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def should_request_more(self, remaining_actions):
        return len(self.buffer) < self.buffer_size or remaining_actions < 3
    
    def add_chunk(self, chunk):
        self.buffer.extend(chunk)
    
    def get_next_action(self):
        return self.buffer.pop(0) if self.buffer else np.zeros(6)
```

### **Option 3: Graceful Degradation**
```python
def safe_network_call(self, openpi_obs, max_retries=2):
    for attempt in range(max_retries):
        try:
            return self.policy_client.infer(openpi_obs)
        except (NetworkTimeout, ConnectionError) as e:
            if attempt == max_retries - 1:
                # Final attempt failed - go to safe position
                print(f"Network failed after {max_retries} attempts: {e}")
                return {"actions": self.get_safe_stop_actions()}
            else:
                print(f"Network attempt {attempt + 1} failed, retrying...")
                time.sleep(0.1)
```

## Implementation Priority

**Phase 1 (Current)**: Get basic functionality working with DROID's approach
- Accept the timeout risk (same as DROID)
- Focus on proving the core concept works
- Manual supervision during testing

**Phase 2 (After First Success)**: Add network resilience
- Implement timeout handling
- Add safe stop behaviors
- Test recovery mechanisms

**Phase 3 (Production)**: Advanced buffering
- Predictive action buffering
- Connection health monitoring
- Comprehensive error recovery

## Testing Strategy for Future Implementation

1. **Simulate Network Failures**: Disconnect network during robot movement
2. **Test Safe Positions**: Verify safe stop actions don't cause collisions
3. **Measure Recovery Time**: Ensure quick recovery to normal operation
4. **Stress Testing**: Long episodes with intermittent network issues

## Notes

- This is a **known limitation** of remote inference architectures
- DROID has the same issue but mitigates through:
  - Local network (lower failure rate)
  - Manual supervision
  - Shorter episodes
- **Starting simple is the right approach** - solve basic problems first
- LeRobot's local inference avoids this entirely (but we want remote inference benefits)

## References

- DROID main.py lines 131-133: Same network call without timeout handling
- OpenPi WebSocket client: Basic implementation without timeout protection
- Risk analysis document: Multi-Module Error Recovery section