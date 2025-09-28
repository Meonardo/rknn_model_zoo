import sys
from rknn.api import RKNN

# model = '../model/hand_sign_v8n.onnx'
model = '../model/hand_sign_v8n.onnx'
DATASET_PATH = '../../../datasets/COCO/coco_sign_14.txt'

RKNN_MODEL = 'hand_sign_v8n_i8.rknn'

rknn_lite = RKNN(verbose=True)
rknn_lite.config(
    target_platform='rk3588', mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]]
)
# mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]]

ret = rknn_lite.load_onnx(model, input_size_list=[[1, 3, 640, 640]])

if ret != 0:
    print('Load model failed!')
    exit(ret)
    
ret = rknn_lite.build(do_quantization=True, dataset=DATASET_PATH)
if ret != 0:
    print('Build model failed!')
    exit(ret)
    
ret = rknn_lite.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export model failed!')
    exit(ret)
    
# # Export CPP demo project
# ret = rknn_lite.codegen(output_path='rknn_app_demo', inputs=['resized_frame.jpg'], overwrite=True)
    
print('Convert onnx to rknn successfully!')

rknn_lite.release()