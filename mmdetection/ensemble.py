import os
import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
import argparse

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='Argparse')

# 입력받을 인자값 설정 (default 값 설정가능)
parser.add_argument('-sp','--sub_path',          type=str,   default='/opt/ml/sample_submission')
parser.add_argument('-t','--ensemble_type',     type=str,   default='wbf')
parser.add_argument('-w','--wbf_weight',     type=int,   nargs='+', default=[])
parser.add_argument('-a','--annotation_file',     type=str, default='test')
parser.add_argument('-i','--iou_thr',     type=float, default=0.6)
parser.add_argument('-g','--sigma',     type=float, default=0.5)
parser.add_argument('-s','--skip_box_thr',     type=float, default=0.0001)
parser.add_argument('-p','--save_path',     type=str, default='ensemble_8_wbf_iou_06_new_dh_fold0_w_2_1')

# args 에 위의 내용 저장
args    = parser.parse_args()

# ensemble csv files
submission_files = os.listdir(args.sub_path)
submission_files = sorted(submission_files)

submission_df = [pd.read_csv(os.path.join(args.sub_path,file)) for file in submission_files]
image_ids = submission_df[0]['image_id'].tolist()

# ensemble 할 file의 image 정보를 불러오기 위한 json
annotation = f'/opt/ml/dataset/{args.annotation_file}.json'
coco = COCO(annotation)

ensemble = args.ensemble_type
prediction_strings = []
file_names = []

# ensemble 시 설정할 iou threshold 이 부분을 바꿔가며 대회 metric에 알맞게 적용해봐요!
iou_thr = args.iou_thr
skip_box_thr = args.skip_box_thr
weights = args.wbf_weight
sigma = args.sigma

# 각 image id 별로 submission file에서 box좌표 추출
for i, image_id in enumerate(image_ids):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]
# 각 submission file 별로 prediction box좌표 불러오기
    for df in submission_df:
        # print(df[df['image_id'] == image_id]['PredictionString'].tolist())
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        
        if len(predict_list)==0 or len(predict_list)==1:
            continue
            
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
            
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))

    if len(boxes_list):
        if args.ensemble_type == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
        elif args.ensemble_type == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif args.ensemble_type == 'softnms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(f'/opt/ml/sample_submission/{args.save_path}.csv')

submission.head()