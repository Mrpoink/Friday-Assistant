from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import shutil

# 1. Download & Export
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Save
save_dir = "onnx_output"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 3. Rename for Android
shutil.move(f"{save_dir}/model.onnx", "all_MiniLM_L6_v2.onnx")
print("File 'all_MiniLM_L6_v2.onnx' is ready. Move it to app/src/main/assets/")