# -*- coding:utf-8 -*-
import sys

sys.path.append('/home/prmct/workspace/caffe-ssd-0103/python')

import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

label_map_file = './coco/labelmap_coco.prototxt'
file = open(label_map_file, 'r')
label_map = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), label_map)


def get_label_name(_labelmap, _labels):
    num_labels = len(_labelmap.item)
    _label_names = []
    if type(_labels) is not list:
        _labels = [_labels]
    for _label in _labels:
        found = False
        for i in xrange(0, num_labels):
            if _label == _labelmap.item[i].label:
                found = True
                _label_names.append(_labelmap.item[i].display_name)
                break
        assert found == True
    return _label_names


def detect_image(_net, _img_pth, conf_threhold=0.3):
    _t1 = datetime.datetime.now()
    image = caffe.io.load_image(_img_pth)
    transformed_image = transformer.preprocess('data', image)
    _net.blobs['data'].data[...] = transformed_image
    _detections = _net.forward()['detection_out']

    # print detections
    det_label = _detections[0, 0, :, 1]
    det_conf = _detections[0, 0, :, 2]
    det_xmin = _detections[0, 0, :, 3]
    det_ymin = _detections[0, 0, :, 4]
    det_xmax = _detections[0, 0, :, 5]
    det_ymax = _detections[0, 0, :, 6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threhold]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_label_name(label_map, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    _t2 = datetime.datetime.now()
    print 'process and detection:', _t2 - _t1

    colors = plt.cm.hsv(np.linspace(0, 1, 80)).tolist()
    currentAxis = plt.gca()
    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    plt.imshow(image)
    plt.title('detection')
    plt.savefig('./save.jpg', dpi=80)
    plt.show()


if __name__ == '__main__':
    work_root = '/home/prmct/Program/detection/ssd/'
    model_def = work_root + 'models/VGGNet/coco/SSD_500x500/deploy.prototxt'
    model_weights = work_root + 'models/VGGNet/coco/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel'

    mean_value = [104, 117, 123]
    img_size = 500

    caffe.set_mode_cpu()
    # caffe.set_device(2)

    t1 = datetime.datetime.now()
    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)
    t2 = datetime.datetime.now()
    print 'load model:', t2 - t1

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array(mean_value))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    net.blobs['data'].reshape(1, 3, img_size, img_size)
    # detect_image(net, '/Users/yanglu/Downloads/n01495701_1192.JPEG')
    detect_image(net, '004545.jpg')
