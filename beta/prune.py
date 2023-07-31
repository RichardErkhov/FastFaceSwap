import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'inswapper_128.onnx'
model_quant = 'inswapper_128.quant.onnx'
#quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic

# Load the original ONNX model
original_model_path = 'inswapper_128.onnx'
quantized_model_path = "inswapper_128.quant.onnx"

# Load the model with ONNX
model = onnx.load(original_model_path)

# Get the last initializer
last_initializer = model.graph.initializer[-1]

# Remove the last initializer from the graph
model.graph.initializer.pop()

# Save the modified model to a temporary file
temp_model_path = "temp_model.onnx"
onnx.save(model, temp_model_path)

# Quantize the model
quantize_dynamic(temp_model_path, quantized_model_path, per_channel=True, weight_type=QuantType.QUInt8)

# Add the last initializer back to the quantized model
quantized_model = onnx.load(quantized_model_path)
quantized_model.graph.initializer.append(last_initializer)

# Save the final quantized model
onnx.save(quantized_model, quantized_model_path)

# Optionally, you can remove the temporary file
import os
os.remove(temp_model_path)