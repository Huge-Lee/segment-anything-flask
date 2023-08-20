import cv2
import requests
import numpy
import json

mask_alpha = 0.3
img_path = '<path/to/image>'  # set your own image path
server_address = 'http://0.0.0.0:2000'


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mask = requests.post(f'{server_address}/points2mask',
                             json.dumps({'points': [[x, y]]})).content
        mask = numpy.frombuffer(mask, numpy.bool8).reshape(img.shape[0], -1)
        mask_int8 = mask.astype(numpy.uint8)*255
        img_cp = img.copy()
        masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask_int8)
        green_img = numpy.zeros_like(img_cp, dtype=numpy.uint8)
        green_img[mask] = [0, 0, 255]
        result = cv2.add(img_cp, cv2.addWeighted(
            masked_img, 1 - mask_alpha, green_img, mask_alpha, 0))
        cv2.imshow('Image', result)


img = cv2.imread(img_path)
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)
img_byte = cv2.imencode('.jpg', img)[1].tobytes()
requests.post(f'{server_address}/setimg', img_byte).content
cv2.imshow("Image", img)
cv2.waitKey(0)
