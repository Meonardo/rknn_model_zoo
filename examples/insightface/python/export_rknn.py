import argparse
import onnx
from rknn.api import RKNN


DATASET_PATH = "../model/dataset/dataset.txt"
TARGET_PLATFORM = "rk3568"

def check_model_input_size(onnx_model_path):
    model = onnx.load(onnx_model_path)
    input_tensor = model.graph.input[0]
    shape = [
        dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim
    ]
    print(f"Model input shape: {shape}")
    return shape


def export_rknn(onnx_model_path, rknn_model_path, input_size, quantized=True):
    rknn_lite = RKNN(verbose=True)
    rknn_lite.config(
        target_platform=TARGET_PLATFORM,
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
    )

    ret = rknn_lite.load_onnx(onnx_model_path, input_size_list=[input_size])

    if ret != 0:
        print('Load model failed!')
        exit(ret)

    ret = rknn_lite.build(do_quantization=quantized, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    ret = rknn_lite.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export model failed!')
        exit(ret)

    print('Convert onnx to rknn successfully!')

    rknn_lite.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify ONNX model input size")
    parser.add_argument("input", help="Path to ONNX model")
    parser.add_argument(
        "--quantized",
        help="Whether to export quantized model (int8)",
    )
    
    args = parser.parse_args()
    
    output_model_path = args.input.replace(".onnx", f"_i8.rknn")
    if not args.quantized:
        output_model_path = args.input.replace(".onnx", f"_f32.rknn")
        
    input_shape = check_model_input_size(args.input)
    export_rknn(args.input, output_model_path, input_shape, quantized=args.quantized)
