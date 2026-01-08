import onnx
import argparse

def check_model_input_size(onnx_model_path):
    model = onnx.load(onnx_model_path)
    input_tensor = model.graph.input[0]
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input tensor shape: {shape}")

    is_batch_dynamic = False

    # check if Batch size is dynamic
    if shape[0] == 0:
        print("Batch size is dynamic.")
        is_batch_dynamic = True
    else:
        print("Batch size is fixed.")
        
    # check if H and W are fixed
    if shape[2] != 0 and shape[3] != 0:
        print("Height and Width are fixed.")
    else:
        print("Height and Width are dynamic.")
        
    return is_batch_dynamic


def update_model_input_size(onnx_model_path, output_model_path, new_size):
    model = onnx.load(onnx_model_path)
    input_tensor = model.graph.input[0]
    for i, dim in enumerate(input_tensor.type.tensor_type.shape.dim):
        # Update H,W dimensions
        if i >= 2:
            dim.dim_value = new_size
    onnx.save(model, output_model_path)
    print(f"Updated model saved to {output_model_path} with new input size {new_size}")


def update_model_batch_size(onnx_model_path, output_model_path, new_batch_size):
    model = onnx.load(onnx_model_path)
    input_tensor = model.graph.input[0]
    for i, dim in enumerate(input_tensor.type.tensor_type.shape.dim):
        # Update batch size dimension
        if i == 0:
            dim.dim_value = new_batch_size
            break
    onnx.save(model, output_model_path)
    print(
        f"Updated model saved to {output_model_path} with new input size {new_batch_size}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify ONNX model input size")
    parser.add_argument("input", help="Path to ONNX model")
    parser.add_argument("--update_size", type=int, default=640, help="New size for H and W dimensions")
    args = parser.parse_args()

    # print the original input size
    is_batch_dynamic = check_model_input_size(args.input)

    # update the model input size
    output_model_path = args.input .replace(".onnx", f"_fixed.onnx")
    if is_batch_dynamic:
        update_model_batch_size(args.input, output_model_path, 1)
    else:
        update_model_input_size(args.input, output_model_path, args.update_size)
    
    # print the updated input size
    check_model_input_size(output_model_path) 
