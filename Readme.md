# Segment-Anything Server with Flask

Lightweight [Segment-Anything](https://github.com/facebookresearch/segment-anything) server for opencv-python base on Flask. In order to use Segment-Anything in no Nvidia GPU computer or embedded devices.

## Server Installation

First install Segment-Anything follow the [official installation guide](https://github.com/facebookresearch/segment-anything#installation) released by Meta Research.

Clone this project locally and install Flask:

```shell
git clone git@github.com:Huge-Lee/segment-anything-flask.git
pip3 install flask
```

Download the checkpoint with: (or you can use the smaller model, don't remember to change the model type by adding startup parameters `[-t model_type]`)

```shell
cd segment-anything-flask/server
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Launch the server and enjoy!

```shell
python3 main.py [-h] [-c checkpoint_path] [-t model_type] [-d device]
```

## How to Use in Client

First, send the origin image to SAM server by using opencv-python:

```python
import cv2
import requests

img = cv2.imread(img_path)
img_byte = cv2.imencode('.jpg', img)[1].tobytes()
requests.post(f'{server_address}/setimg', img_byte).content
```

Then, send points to SAM server and you will get the mask (type: numpy.bool8) :

```python
import numpy
import json

raw_mask = requests.post(f'{server_address}/points2mask',
                     json.dumps({'points':[[x1, y1],[x2, y2]]})).content
mask = numpy.frombuffer(raw_mask, numpy.bool8).reshape(img.shape[0], -1)
```

You can run a mouse clicked example in `examples/click_demo.py` (you should set a right path to a image in the python file).