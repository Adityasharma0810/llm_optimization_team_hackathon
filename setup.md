# Setup Guide

This file is the shortest end-to-end path for using a shared AWS EC2 Arm instance
with this project.

It assumes:
- Your friend already created the EC2 instance
- You have the `.pem` file for SSH access
- The instance is running Ubuntu on ARM64
- You want to run the dashboard from your Windows machine and use the EC2 as the
  llama-server backend

## Backend On EC2

### 1. SSH into the EC2 instance

From PowerShell on Windows:

```powershell
icacls.exe ".\lulu.pem" /inheritance:r /grant:r "$env:USERNAME:(R)"
ssh -i ".\lulu.pem" ubuntu@3.107.4.17
```

If `ubuntu` does not work, ask your friend which Linux username the AMI uses.

### 2. Clone the repository on the EC2 instance

Run these on the SSH session:

```bash
cd ~
git clone https://github.com/Adityasharma0810/llm_optimization_team_hackathon.git
cd llm_optimization_team_hackathon
```

### 3. Build llama.cpp on the EC2 instance

The repo provides `setup.sh`, but on Ubuntu 26.04 the CMake version check may
need the fixed version already used during setup.

Run:

```bash
chmod +x setup.sh
./setup.sh -j 2
```

If the script ever stops on the CMake version check, the underlying issue is
the script comparing versions incorrectly. The build itself still works.

### 4. Verify the binary exists

After setup finishes:

```bash
ls ~/llama.cpp/build/bin/llama-server
```

If that file exists, the server binary is ready.

### 5. Download a GGUF model

Create a download folder:

```bash
mkdir -p /home/ubuntu/quantized
cd /home/ubuntu/quantized
```

Install the Hugging Face helper in system Python:

```bash
python3 -m pip install --break-system-packages -U huggingface_hub
```

Download the model:

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='/home/ubuntu/quantized', local_dir_use_symlinks=False)"
```

If your friend uses a different model, replace the repo and filename with the
actual GGUF they want to serve.

### 6. Start llama-server

Run this on the EC2 instance:

```bash
~/llama.cpp/build/bin/llama-server \
  -m /home/ubuntu/quantized/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080
```

Keep this terminal open while the server is running.

### 7. Verify the server on the EC2 instance

Open a second SSH session to the same instance and run:

```bash
curl http://localhost:8080/health
```

Expected result:

```json
{"status":"ok"}
```

### 8. Verify the server from your Windows machine

Run this from PowerShell:

```powershell
curl.exe http://3.107.4.17:8080/health
```

If this fails but `localhost:8080` works on the EC2 instance, the AWS security
group probably does not allow inbound port `8080`.

## Frontend On Windows

### 9. Start the frontend locally

From your Windows machine:

```powershell
cd C:\Users\adira\llm_optimization_team_hackathon
$env:SPECARM_SERVER_URL="http://3.107.4.17:8080"
python dashboard\app.py
```

Open:

```text
http://localhost:5000
```

If you prefer to keep the frontend pointed at a different backend later,
change `SPECARM_SERVER_URL` to that server's public URL.

### 10. What the frontend is doing

The dashboard is a local Flask app that:
- serves the web UI from `dashboard/index.html`
- checks the backend health endpoint at `/health`
- sends benchmark and prompt requests to `/completion`

That means the frontend only works when the EC2 server is reachable.
If the page says "Server Offline", it usually means the EC2 instance is not
reachable on port `8080`.

### 11. Run benchmarks

From the repo root on Windows:

```powershell
python -m benchmark.benchmark --server-url http://3.107.4.17:8080 --model-name Q4_K_M --trials 3 --max-tokens 64
```

Or use the dashboard Benchmark tab and keep the same server URL.

## Notes

- Do not commit `.pem` files or model files to git.
- The EC2 public IP can change if the instance is stopped and started again
  unless an Elastic IP is attached.
- The repo’s longer guide is still available in `docs/SETUP.md`.
