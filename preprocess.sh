echo "preprocess"
python3 getOnnx.py
echo "get onnx done"
python3 onnx2trt.py
