# check target onnx model 
# print model information: input and output names, shapes, and types
import argparse
import onnx
import os


def check_onnx_model(model_path):
    """
    Check the ONNX model for input and output details.
    
    :param model_path: Path to the ONNX model file.
    """
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        print(f"Model '{model_path}' is valid.")
        print("Inputs:")
        for input in model.graph.input:
            print(f"  Name: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}, Type: {input.type.tensor_type.elem_type}")
        
        print("Outputs:")
        for output in model.graph.output:
            print(f"  Name: {output.name}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}, Type: {output.type.tensor_type.elem_type}")
    
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check ONNX Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the ONNX model file')
    
    args = parser.parse_args()
    
    # Check if the model path exists
    if not os.path.exists(args.model_path):
        print(f"Model path '{args.model_path}' does not exist.")
        exit(1)
    
    # Check the ONNX model
    check_onnx_model(args.model_path)
    print("ONNX model check completed.")