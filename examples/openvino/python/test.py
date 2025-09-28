import numpy as np
import cv2
from rknn.api import RKNN

RKNN_MODEL_PATH = 'student_action_recognition.rknn'
TEST_VIDEO_PATH = 'student_10s.mp4'
TEST_VIDEO_OUT_PATH = 'test_output.mp4'

# Model input size (NCHW): [1, 3, 400, 680]
INPUT_W = 680
INPUT_H = 400

# The image size of the original video(updated in code)
IMG_W = 1920
IMG_H = 1080

DETECTION_CONFIDENCE_THRESHOLD = 0.3
ACTION_CONFIDENCE_THRESHOLD = 0.75
ACTION_SCALE = 3.0
NUM_ACTION_CLASSES = 3
ACTION_CLASS_LIST = [
    'sitting', 'standing', 'raising_hand'
]
ACTION_COLOR_LIST = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255)
]

VARIANCES = [0.1, 0.1, 0.2, 0.2]
# Head/anchor info
# Width: 29.9271, 41.4375, 58.0833, 91.0208
# Height: 67.037, 90.3704, 129.259, 190
HEADS = {
    'step': 16, 
    'anchors': [
        (29.9271, 67.037),
        (41.4375, 90.3704),
        (58.0833, 129.259),
        (91.0208, 190)
    ]
}
KEEP_TOP_K = 200
NMS_SIGMA = 0.6

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def parse_bbox_record(data):
    return [data[0], data[1], data[2], data[3]]

def convert_to_rect(prior_bbox, variances, encoded_bbox, frame_size):
    prior_width = prior_bbox[2] - prior_bbox[0]
    prior_height = prior_bbox[3] - prior_bbox[1]
    prior_center_x = 0.5 * (prior_bbox[0] + prior_bbox[2])
    prior_center_y = 0.5 * (prior_bbox[1] + prior_bbox[3])

    decoded_bbox_center_x = variances[0] * encoded_bbox[0] * prior_width + prior_center_x
    decoded_bbox_center_y = variances[1] * encoded_bbox[1] * prior_height + prior_center_y
    decoded_bbox_width = np.exp(variances[2] * encoded_bbox[2]) * prior_width
    decoded_bbox_height = np.exp(variances[3] * encoded_bbox[3]) * prior_height

    decoded_bbox_xmin = decoded_bbox_center_x - 0.5 * decoded_bbox_width
    decoded_bbox_ymin = decoded_bbox_center_y - 0.5 * decoded_bbox_height
    decoded_bbox_xmax = decoded_bbox_center_x + 0.5 * decoded_bbox_width
    decoded_bbox_ymax = decoded_bbox_center_y + 0.5 * decoded_bbox_height

    x1 = int(decoded_bbox_xmin * frame_size[0])
    y1 = int(decoded_bbox_ymin * frame_size[1])
    x2 = int(decoded_bbox_xmax * frame_size[0])
    y2 = int(decoded_bbox_ymax * frame_size[1])
    return (x1, y1, x2, y2)

def soft_nms(detections, sigma, top_k, min_det_conf):
    scores = np.array([d['detection_conf'] for d in detections])
    n = len(scores)
    top_k = min(top_k, n)
    score_idx = np.arange(n)
    if n > 0 and top_k > 0:
        if top_k < n:
            score_idx = score_idx[np.argpartition(-scores, top_k-1)[:top_k]]
    else:
        return []
    top_scores = scores[score_idx].copy()
    out_indices = []
    for _ in range(len(top_scores)):
        best_idx = np.argmax(top_scores)
        if top_scores[best_idx] < min_det_conf:
            break
        anchor_idx = score_idx[best_idx]
        out_indices.append(anchor_idx)
        top_scores[best_idx] = 0.0
        rect1 = detections[anchor_idx]['rect']
        for i, ref_idx in enumerate(score_idx):
            if top_scores[i] < min_det_conf:
                continue
            rect2 = detections[ref_idx]['rect']
            xA = max(rect1[0], rect2[0])
            yA = max(rect1[1], rect2[1])
            xB = min(rect1[2], rect2[2])
            yB = min(rect1[3], rect2[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            area1 = max(1, (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]))
            area2 = max(1, (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]))
            iou = interArea / float(area1 + area2 - interArea)
            top_scores[i] *= np.exp(-iou * iou / sigma)
    return out_indices

def generate_prior_box(pos, step, anchor, blob_size):
    row = pos // blob_size[0]
    col = pos % blob_size[0]
    center_x = (col + 0.5) * step
    center_y = (row + 0.5) * step
    xmin = (center_x - 0.5 * anchor[0]) / INPUT_W
    ymin = (center_y - 0.5 * anchor[1]) / INPUT_H
    xmax = (center_x + 0.5 * anchor[0]) / INPUT_W
    ymax = (center_y + 0.5 * anchor[1]) / INPUT_H
    return [xmin, ymin, xmax, ymax]

if __name__ == "__main__":
    rknn = RKNN(verbose=True)

    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Load RKNN model \"{}\" failed!'.format(RKNN_MODEL_PATH))
        exit(ret)

    print('Load rknn model success!')

    ret = rknn.init_runtime(target='rk3568')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)

    print('Init runtime environment success!') 
   
    ########################################################################################
    # D RKNNAPI: Input tensors:
    # D RKNNAPI:   index=0, name=images, n_dims=4, dims=[1, 400, 680, 3], n_elems=816000, size=1632000, 
    # D RKNNAPI: Output tensors:
    # D RKNNAPI:   index=0, name=bboxes, n_dims=2, dims=[1, 17200], n_elems=17200, size=34400
    # D RKNNAPI:   index=1, name=bboxes_scores, n_dims=2, dims=[1, 8600], n_elems=8600, size=17200
    # D RKNNAPI:   index=2, name=anchor3, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
    # D RKNNAPI:   index=3, name=anchor2, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
    # D RKNNAPI:   index=4, name=anchor1, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
    # D RKNNAPI:   index=5, name=anchor4, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
    
    # Input:
    # Name: images, Shape: [1, 3, 400, 680], Type: 1 (NCHW, BGR)
    # Outputs:
    # Name: bboxes, Shape: [1, 17200], Type: 1
    # Name: priorbox, Shape: [1, 2, 17200], Type: 1
    # Name: bboxes_scores, Shape: [1, 8600], Type: 1
    # Name: anchor3, Shape: [1, 25, 43, 3], Type: 1 (NHWC)
    # Name: anchor2, Shape: [1, 25, 43, 3], Type: 1 (NHWC)
    # Name: anchor1, Shape: [1, 25, 43, 3], Type: 1 (NHWC)
    # Name: anchor4, Shape: [1, 25, 43, 3], Type: 1 (NHWC)
    ########################################################################################

    # Input video 
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    IMG_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    IMG_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(TEST_VIDEO_OUT_PATH, fourcc, cap.get(cv2.CAP_PROP_FPS), (IMG_W, IMG_H))

    output_names = []
    print("Model output names:", output_names)
    # Match output names by substring
    loc_name = 'bboxes'
    conf_name = 'bboxes_scores'
    priorbox_name = 'priorboxes'
    anchor_names = [ # order matters
        'anchor1',
        'anchor2',
        'anchor3',
        'anchor4'
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # NCHW input
        frame_resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        frame_input = frame_resized.transpose(2, 0, 1)  # HWC -> CHW
        frame_input = frame_input[None, ...].astype('float32')  # Add batch dim

        # result is ndarray list
        results = rknn.inference(inputs=[frame_input], data_format='nchw')

        # Box coordinates in SSD format
        loc_raw = results[output_names.index(loc_name)][0]  # [1, 17200]
        loc_raw = np.squeeze(loc_raw)  # [17200]
        conf_raw = results[output_names.index(conf_name)][0]  # [1, 8600]
        conf_raw = np.squeeze(conf_raw)  # [8600]
        priorbox_raw = results[output_names.index(priorbox_name)][0]  # [1, 2, 17200]
        priorbox_raw = np.squeeze(priorbox_raw)  # [2, 17200]
        priorbox_raw = priorbox_raw.flatten()  # [34400,]
        anchor_confs = [np.squeeze(results[output_names.index(name)][0]) for name in anchor_names]  # [1, 25, 43, 3] -> [25, 43, 3]
        if not anchor_confs:
            print("No anchor outputs found!")
            continue

        num_candidates = min(
            loc_raw.shape[0] // 4,
            conf_raw.shape[0] // 2,
            priorbox_raw.shape[0] // 4
        )
        detections = []

        # priorbox shape: [1, 2, num_candidates*2]
        for p in range(num_candidates):
            # Parse detection confidence from the SSD Detection output
            detection_conf = conf_raw[2*p+1] 
            # Skip low-confidence detections
            if detection_conf < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            # Estimate the action anchor ID
            anchor_id = p % 4
            
            # Estimate the action label
            anchor_conf_blob = anchor_confs[anchor_id]  # [25, 43, 3]
            # Flatten to 1d array
            anchor_flat = anchor_conf_blob.ravel()  # [25*43*3,]

            anchor_idx = p // 4 * NUM_ACTION_CLASSES
            if anchor_idx >= len(anchor_flat):
                continue
            
            action_label = -1
            action_max_exp_value = 0.0
            action_sum_exp_values = 0.0
            for c in range(NUM_ACTION_CLASSES):
                action_exp_value = np.exp(anchor_flat[anchor_idx + c] * ACTION_SCALE)
                action_sum_exp_values += action_exp_value
                if action_exp_value > action_max_exp_value:
                    action_max_exp_value = action_exp_value
                    action_label = c

            # action_sum_exp_values can't be equal to 0
            if abs(action_sum_exp_values) < 1e-6:
                action_label = 0
                action_conf = 0.0
            # Estimate the action confidence
            action_conf = action_max_exp_value / action_sum_exp_values
            
            # Skip low-confidence actions    
            if action_label < 0 or action_conf < ACTION_CONFIDENCE_THRESHOLD:
                action_label = 0
                action_conf = 0.0

            # Priorbox extraction
            # xmin, ymin, xmax, ymax
            prior_bbox = generate_prior_box(p // 4, HEADS['step'], HEADS['anchors'][anchor_id], (43, 25))
            print("Prior box:", prior_bbox)
            # Variance are constants
            # Box coordinates
            encoded_bbox = loc_raw[4*p : 4*p+4]

            # Convert to the input image coordinates
            rect = convert_to_rect(prior_bbox, VARIANCES, encoded_bbox, (IMG_W, IMG_H))

            detections.append({
                'rect': rect,
                'label': action_label,
                'detection_conf': detection_conf,
                'action_conf': action_conf
            })

        print("Detection before NMS: ", len(detections))
        # Soft-NMS
        out_indices = soft_nms(detections, NMS_SIGMA, KEEP_TOP_K, DETECTION_CONFIDENCE_THRESHOLD)
        final_detections = [detections[i] for i in out_indices]
        
        print("Detection after NMS: ", len(final_detections))

        # Draw detections
        for det in final_detections:
            x1, y1, x2, y2 = det['rect']
            label = det['label']
            det_conf = det['detection_conf']
            act_conf = det['action_conf']

            bbox_color = (255, 255, 255) # white color
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 1)
            
            action_color = ACTION_COLOR_LIST[label] if 0 <= label < len(ACTION_COLOR_LIST) else (255, 255, 255)
            # text = f"{ACTION_CLASS_LIST[label]} ({det_conf:.2f}/{act_conf:.2f})"
            text = f"{ACTION_CLASS_LIST[label]}"
            # draw background color for the text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            base_line = text_size[1]
            text_size = text_size[0]
            cv2.rectangle(frame, (x1, max(0, y1-text_size[1])), (x1 + text_size[0], y1+base_line), bbox_color, -1)
            # text
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, action_color, 1, lineType=cv2.LINE_AA)

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"Annotated video saved to {TEST_VIDEO_OUT_PATH}")