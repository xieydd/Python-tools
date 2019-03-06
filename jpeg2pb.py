# -*- coding: utf-8 -*-


from PIL import Image
import numpy as np
import onnx
from onnx import numpy_helper

# jpeg for yolo 416x416 CHW
def jpeg2pb(jpeg_path, pb_path):
    img = Image.open(jpeg_path)
    trans = img.resize((416,416), Image.ANTIALIAS)
    data = np.asarray(trans, dtype=np.float32)
    tensor = numpy_helper.from_array(data.transpose(2,0,1).reshape(1,3,416,416))
    with open(pb_path, 'wb') as f:
        f,write(tensor.SerializeToString())

def pb2array(pb_path):
    tensor = onnx.TensorProto()
    with open(pb_path) as f:
        tensor.ParseFromString(f.read())
    return numpy_helper.to_array(tensor)
