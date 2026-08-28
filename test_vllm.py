import subprocess
try:
    subprocess.check_output(
        "vllm serve models/Mistral-7B-Instruct-v0.3 --quantization bitsandbytes --load-format bitsandbytes --max-model-len 1024 & PID=$! && sleep 10 && kill $PID",
        shell=True, stderr=subprocess.STDOUT
    )
    print("SUCCESS")
except subprocess.CalledProcessError as e:
    print(e.output.decode())
