# -*- coding:utf-8 -*-
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('/Users/yanglu/workspace/caffe-master-0503/python')

import caffe
import cv2
import numpy as np
import datetime

OUTPUT_LAYER = 'dconv3'
NORMALIZATION = 1.0  # image scale factor
mean_value = np.array([103.939, 116.779, 123.68])


def net_predict(net_0, _img):
    _img = np.asarray(_img, dtype=np.float32)
    _img -= mean_value
    _img = _img.transpose(2, 0, 1)
    _img = _img.reshape((1,) + _img.shape)

    net_0.blobs['data'].reshape(*_img.shape)
    net_0.blobs['data'].data[...] = _img / NORMALIZATION
    net_0.forward()
    output_prob = net_0.blobs[OUTPUT_LAYER].data
    # output_prob[0][0] *= 130
    # output_prob[0][1] *= 130
    # output_prob[0][2] *= 130
    return output_prob * 130


if __name__ == "__main__":
    caffe.set_mode_cpu()
    # caffe.set_device(0)  # just for GPU mode
    net = caffe.Net('./caffe_style.prototxt', './feathers_b8_es.caffemodel', caffe.TEST)

    content_img = cv2.imread('./ins.jpg')  # image read

    s_time = datetime.datetime.now()
    synthesis_img = net_predict(net, content_img).transpose(0, 2, 3, 1)  # net forward
    print 'time cost:', datetime.datetime.now() - s_time

    synthesis_img = synthesis_img.reshape((synthesis_img.shape[1:]))  # post-process
    synthesis_img = synthesis_img + mean_value
    synthesis_img = np.maximum(0, synthesis_img)
    synthesis_img = np.uint8(synthesis_img)
    synthesis_img = cv2.medianBlur(synthesis_img, 3)  # median clur

    cv2.imwrite('./synthesis.jpg', synthesis_img)  # image write
    cv2.imshow('content', content_img)
    cv2.imshow('synthesis', synthesis_img)
    cv2.waitKey()

