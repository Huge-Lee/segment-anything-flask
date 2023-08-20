from core import Flask_app, request, jsonify
import numpy
import cv2
import json


@Flask_app.route('/setimg', methods=['POST'])
def set_img():
    try:
        img = request.data
        img = numpy.frombuffer(img, numpy.uint8)
        img = cv2.imdecode(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        Flask_app.predictor.set_image(img)
    except:
        return jsonify({'status': "failed"})
    return jsonify({'status': "success"})


@Flask_app.route('/points2mask', methods=['POST'])
def points_to_mask():
    points = json.loads(request.data)['points']

    masks, _, _ = Flask_app.predictor.predict(
        point_coords=numpy.array(points),
        point_labels=numpy.array([1 for _ in range(len(points))]),
        multimask_output=False,
    )

    return masks.tobytes()
