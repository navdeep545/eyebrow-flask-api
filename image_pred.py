import argparse

import numpy as np
import cv2

import torch
import torchvision

from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces

from shapely.geometry import Polygon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

	checkpoint = torch.load(args.model_path, map_location=device)
	pfld_backbone = PFLDInference().to(device)
	pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
	pfld_backbone.eval()
	pfld_backbone = pfld_backbone.to(device)
	transform = torchvision.transforms.Compose(
		[torchvision.transforms.ToTensor()])


	filename = input("Please enter filename: ")
	test_face = cv2.imread(filename)

	# Just for debugging purposes:
	img = test_face.copy()


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


		

		#new_landmark = convert_points(pre_landmark.copy(), x , y)

		right_point = pre_landmark.astype(np.int32)[33:42]
		left_point = pre_landmark.astype(np.int32)[42:51]
		all_points =  pre_landmark.astype(np.int32)


		points_left_n = translate_eyebrow2(right_point, left_point, all_points)



		# right brow 
		for (x, y) in pre_landmark.astype(np.int32)[33:42]:
			cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))

		# left brow 
		for (x, y) in pre_landmark.astype(np.int32)[42:51]:
			cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))

		# new left brow 
		for (x, y) in points_left_n.astype(np.int32):
			cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255))


		#cv2.polylines(img, [points.astype(np.int32)], True, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
		new_left_points = convert_points(points_left_n.astype(np.int32),x1,y1)

		#cv2.drawContours(img, [new_points.astype(np.int32)], 0, 255, -1)


		right_area = cv2.contourArea(right_point)
		print("right_area: ", right_area)
		left_area = cv2.contourArea(left_point)
		print("left area: ", left_area)

		intersection_area = find_interestion(right_point, points_left_n)

		print("IOU Score: ", intersection_area/(right_area + left_area))



	cv2.imshow('face_landmark_68', img)
	if cv2.waitKey(0) == 27:
		cv2.destroyAllWindows()


# def translate_eyebrow(right_point, left_point, midpoint):
#     dist_rpf_lpf_x = np.abs(left_point[0][0] - midpoint[0])

#     new_left_point = []#
#     for i in range(len(left_point)):
#         d1 = dist_rpf_lpf_x
#         d2 = np.abs(left_point[i][0] - midpoint[0])
#         l_2nd_new = np.abs([left_point[i][0] - 2*(d1+d2), left_point[i][1]])
		
#         new_left_point.append(l_2nd_new)
#     points = np.array(new_left_point, dtype=np.int32)
#     #print("points: ", points)
#     #cv2.polylines(image, [points], is_closed, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)

#     return points 

def find_interestion(point1, point2):
	

	# Create two Polygon objects
	poly1 = Polygon(point1)
	poly2 = Polygon(point2)

	# Find the intersection of the two polygons
	intersection_poly = poly1.intersection(poly2)

	# Calculate the area of the intersection
	intersection_area = intersection_poly.area

	print("Intersection area:", intersection_area)

	return intersection_area


def convert_points(points, x , y):
	new_points = []
	for (a, b) in points: 
		x2 = a+x
		y2 = b+y 

		new_points.append([x2, y2])
	return np.array(new_points).astype(np.int32)



def translate_eyebrow2(right_point, left_point, all_points):

	dist_rpf_lpf_x = np.abs(all_points[37][0] - all_points[42][0])

	new_left_point = []
	for i in range(len(left_point)):
		if i ==0: 
			# first point left 
			l_1st_new = np.abs([left_point[i][0] - dist_rpf_lpf_x, left_point[i][1]])
			new_left_point.append(l_1st_new)
		else: 

			# 2nd point left 
			d1 = dist_rpf_lpf_x
			d2 = (left_point[i][0] - all_points[42][0])
			l_2nd_new = np.abs([left_point[i][0] - (d1+2*d2), left_point[i][1]])

			new_left_point.append(l_2nd_new)
	points = np.array(new_left_point, dtype=np.int32)

	return points 


def parse_args():
	parser = argparse.ArgumentParser(description='Testing')
	parser.add_argument('--model_path',
						default="./checkpoint/snapshot/checkpoint.pth.tar",
						type=str)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	main(args)
