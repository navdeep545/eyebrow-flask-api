import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import jsonpickle
import cv2
import torch
import torchvision
import base64
from pfld import PFLDInference, AuxiliaryNet
from detector import detect_faces
from shapely.geometry import Polygon

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filename):
    checkpoint = torch.load("checkpoint.pth.tar", map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    img = cv2.imread(filename)
    height, width = img.shape[:2]
    bounding_boxes, landmarks = detect_faces(img)
    for box in bounding_boxes:
        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2

        size = int(max([w, h]) * 1.1)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - width)
        edy2 = max(0, y2 - height)

        cropped = img[y1:y2, x1:x2]
        if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                         cv2.BORDER_CONSTANT, 0)

        input_ = cv2.resize(cropped, (112, 112))
        input_ = transform(input_).unsqueeze(0).to(device)
        _, landmarks = pfld_backbone(input_)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
            -1, 2) * [size, size] - [edx1, edy1]

        right_point = pre_landmark.astype(np.int32)[33:42]
        left_point = pre_landmark.astype(np.int32)[42:51]
        all_points =  pre_landmark.astype(np.int32)

        points_left_n = translate_eyebrow2(right_point, left_point, all_points)

        new_left_points = convert_points(points_left_n.astype(np.int32),x1,y1)

        right_area = cv2.contourArea(right_point)
        left_area = cv2.contourArea(left_point)
        intersection_area = find_interestion(right_point, points_left_n)

        iou_score = intersection_area / (right_area + left_area)

        # Draw landmarks on the image
        for (x, y) in right_point:
            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))
        for (x, y) in left_point:
            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))
        for (x, y) in points_left_n:
            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255))

        # Encode image to base64
        _, encoded_img = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(encoded_img).decode('utf-8')

    return iou_score, img_base64

def translate_eyebrow2(right_point, left_point, all_points):
    dist_rpf_lpf_x = np.abs(all_points[37][0] - all_points[42][0])

    new_left_point = []
    for i in range(len(left_point)):
        if i ==0: 
            l_1st_new = np.abs([left_point[i][0] - dist_rpf_lpf_x, left_point[i][1]])
            new_left_point.append(l_1st_new)
        else: 
            d1 = dist_rpf_lpf_x
            d2 = (left_point[i][0] - all_points[42][0])
            l_2nd_new = np.abs([left_point[i][0] - (d1+2*d2), left_point[i][1]])
            new_left_point.append(l_2nd_new)
    points = np.array(new_left_point, dtype=np.int32)
    return points 

def find_interestion(point1, point2):
    poly1 = Polygon(point1)
    poly2 = Polygon(point2)
    intersection_poly = poly1.intersection(poly2)
    intersection_area = intersection_poly.area
    return intersection_area

def convert_points(points, x , y):
    new_points = []
    for (a, b) in points: 
        x2 = a+x
        y2 = b+y 
        new_points.append([x2, y2])
    return np.array(new_points).astype(np.int32)

@app.route('/media/upload', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    # if file.filename == '':
        # return jsonify({'error': 'no file selected'}), 400
    # if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)
    iou_score, img_base64 = process_file(file)
    return jsonify({'msg': 'media uploaded successfully', 'iou_score': iou_score, 'processed_image': img_base64})
    # return jsonify({'error': 'file type not allowed'}), 400
    # r = request
    # convert string of image data to uint8
    # nparr = np.fromstring(r.data, np.uint8)
    # decode image
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # iou_score, img_base64 = process_file(img)
    # do some fancy processing here....


    # build a response dict to send back to client
    # response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                # }
    # encode response using jsonpickle
    # response_pickled = jsonpickle.encode(response)

    # return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
