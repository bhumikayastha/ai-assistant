from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./onnx_model")
tokenizer.save_pretrained("./onnx_model")

print("ONNX model saved to ./onnx_model")