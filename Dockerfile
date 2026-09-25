# Ultralytics official image comes with CUDA, PyTorch, OpenCV and YOLO pre-installed
FROM ultralytics/ultralytics:latest

# Set working directory inside container
WORKDIR /workspace

# Copy local training files into container
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/

# Default training command
CMD ["python3", "train.py", "--epochs", "100", "--batch", "16", "--device", "0"]
