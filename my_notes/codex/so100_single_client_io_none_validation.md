SO100 Single-Arm: IO-None Validation Steps

1) Start the policy server (on the AutoDL machine)
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
   uv run scripts/serve_policy.py policy:checkpoint \
     --policy.config=pi0_clear_tray_fine_tune \
     --policy.dir=/root/autodl-tmp/openpi/checkpoints/pi0_clear_tray_fine_tune/my_first_pi0jax_run/15000 \
     --port 8000

2) If connecting from a PC, create an SSH tunnel (on the PC)
   ssh -p <SSH_PORT> -N -L 8000:<CONTAINER_IP>:8000 <USER>@<SERVER_PUBLIC_IP>
   # Example: ssh -p 55778 -N -L 8000:172.17.0.3:8000 root@connect.bjc1.seetacloud.com

3) Run the SO100 single client with no robot I/O (camera/robot disabled)
   uv run examples/so100_single_client/main.py \
     --host <SERVER_IP_OR_localhost> \
     --port 8000 \
     --io-mode none \
     --freq-hz 5 \
     --prompt "clear the tray"

4) Verify
   - Client connects and prints timing stats without errors.
   - Server logs requests and returns actions (check server_timing in responses).

