import sys

sys.path.append('/home/prmct/workspace/PSPNet-1212/python/')

import caffe
import cv2
import numpy as np
import datetime

gpu_mode = True
gpu_id = 0
data_root = '/home/gaia/Dataset/VOC_PASCAL/VOC2012_test/JPEGImages/'
val_file = 'test.txt'
save_root = './ss/'
model_weights = 'pspnet101_VOC2012.caffemodel'
model_deploy = 'pspnet101_VOC2012_473.prototxt'
prob_layer = 'prob'  # output layer, normally Softmax
class_num = 21
dynamic_edge = [32 * i + 1 for i in xrange(15)]
raw_scale = 1.0  # image scale factor, 1.0 or 128.0
mean_value = np.array([104.008, 116.669, 122.675])
# scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]   # multi scale
scale_array = [1.0]  # single scale
flip = True
class_offset = 0

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)


def eval_batch():
    eval_images = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip())

    skip_num = 0
    eval_len = len(eval_images)
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = cv2.imread(data_root + eval_images[i + skip_num] + '.jpg')
        h, w, d = _img.shape

        score_map = np.zeros((h, w, class_num), dtype=np.float32)
        for j in scale_array:
            _scale = size_regularization(_img, scale=j)
            score_map += cv2.resize(caffe_process(_scale - mean_value), (w, h))
        score_map /= len(scale_array)
        cv2.imwrite(save_root + eval_images[i + skip_num] + '.png', score_map.argmax(2) + class_offset)
        print 'Testing image: ' + str(i + 1) + '/' + str(eval_len) + '  ' + str(eval_images[i + skip_num])
    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    print '\n{} images has been tested. \nThe model is: {}'.format(str(eval_len), model_weights)


def size_regularization(_img, scale=1.0):
    h, w, d = _img.shape
    ratio = float(dynamic_edge[-1]) / max(h, w)
    _tmp = cv2.resize(_img, (int(w * ratio * scale), int(h * ratio * scale)))
    target_size = ([x for x in dynamic_edge if _tmp.shape[1] - x >= 0][-1],
                   [x for x in dynamic_edge if _tmp.shape[0] - x >= 0][-1])

    return cv2.resize(_img, target_size)


def caffe_process(_input):
    h, w, d = _input.shape
    _score = np.zeros((h, w, class_num), dtype=np.float32)
    if flip:
        _flip = _input[:, ::-1]
        _flip = _flip.transpose(2, 0, 1)
        _flip = _flip.reshape((1,) + _flip.shape)
        net.blobs['data'].reshape(*_flip.shape)
        net.blobs['data'].data[...] = _flip / raw_scale
        net.blobs['data_dim'].data[...] = [[[h, w]]]
        net.forward()
        _score += net.blobs[prob_layer].data[0].transpose(1, 2, 0)[:, ::-1]

    _input = _input.transpose(2, 0, 1)
    _input = _input.reshape((1,) + _input.shape)
    net.blobs['data'].reshape(*_input.shape)
    net.blobs['data'].data[...] = _input / raw_scale
    net.blobs['data_dim'].data[...] = [[[h, w]]]
    net.forward()
    _score += net.blobs[prob_layer].data[0].transpose(1, 2, 0)

    return _score / int(flip + 1)

if __name__ == '__main__':
    eval_batch()

