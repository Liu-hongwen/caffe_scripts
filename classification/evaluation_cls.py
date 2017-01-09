import sys

sys.path.append('/home/prmct/workspace/caffe-master-1202/python')

import numpy as np
import caffe
import cv2
import datetime

gpu_mode = True
gpu_id = 3
data_root = '/home/prmct/Database/ILSVRC2016'
val_file = 'resnet101_v2/ILSVRC2012sub_val.txt'
save_log = 'log{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
model_weights = 'resnet269_v2/resnet269_v2.caffemodel'
model_deploy = 'resnet269_v2/deploy_resnet269_v2.prototxt'
prob_layer = 'prob'
class_num = 1000
base_size = 256
crop_size = 224
raw_scale = 1.0
mean_value = np.array([128, 128, 128])
top_k = (1, 5)
class_offset = 0
crop_num = 1  # 1 and others for single-crop, 12 for 12-crop, 144 for 144-crop

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)


def eval_batch():
    eval_images = []
    ground_truth = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip().split(' ')[0])
        ground_truth.append(int(i.strip().split(' ')[1]))
    f.close()

    skip_num = 0
    eval_len = len(eval_images)
    accuracy = np.zeros(len(top_k))
    # eval_len = 100
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = cv2.imread(data_root + eval_images[i + skip_num])

        score_vec = np.zeros(class_num, dtype=np.float32)
        crops = []
        if crop_num == 1:
            crops.append(cv2.resize(_img, (crop_size, crop_size)))
        elif crop_num == 12:
            crops.extend(mirror_crop(_img))
        elif crop_num == 144:
            crops.extend(multi_crop(_img))
        else:
            crops.append(cv2.resize(_img, (crop_size, crop_size)))

        for j in crops:
            score_vec += caffe_process(j)
        score_index = (-score_vec).argsort()

        print 'Testing image: ' + str(i + 1) + '/' + str(eval_len) + '  ' + str(score_index[0]) + '/' + str(
            ground_truth[i + skip_num]),
        for j in xrange(len(top_k)):
            if ground_truth[i + skip_num] in score_index[:top_k[j]]:
                accuracy[j] += 1
            tmp_acc = float(accuracy[j]) / float(i + 1)
            if top_k[j] == 1:
                print '\ttop_' + str(top_k[j]) + ':' + str(tmp_acc),
            else:
                print 'top_' + str(top_k[j]) + ':' + str(tmp_acc)

    end_time = datetime.datetime.now()
    w = open(save_log, 'w')
    s1 = 'Evaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    s2 = '\n{} images has been tested, crop_num is: {}, crop_size is: {}. \nThe model is: {}'.format(str(eval_len),
                                                                                                     str(crop_num),
                                                                                                     str(crop_size),
                                                                                                     model_weights)
    s3 = ''
    for i in xrange(len(top_k)):
        _acc = float(accuracy[i]) / float(eval_len)
        s3 += '\nAccuracy of top_{} is: {}; correct num is {}.'.format(str(top_k[i]), str(_acc), str(int(accuracy[i])))
    print s1, s2, s3
    w.write(s1 + s2 + s3)
    w.close()


def over_sample(img):  # 12 crops of image
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    sample_list = [img[:crop_size, :crop_size], img[-crop_size:, -crop_size:], img[:crop_size, -crop_size:],
                   img[-crop_size:, :crop_size], img[yy: yy + crop_size, xx: xx + crop_size],
                   cv2.resize(img, (crop_size, crop_size))]
    return sample_list


def mirror_crop(img):  # 12*len(size_list) crops
    crop_list = []
    img_resize = cv2.resize(img, (base_size, base_size))
    mirror = img_resize[:, ::-1]
    crop_list.extend(over_sample(img_resize))
    crop_list.extend(over_sample(mirror))
    return crop_list


def multi_crop(img):  # 144(12*12) crops
    crop_list = []
    size_list = [256, 288, 320, 352]  # crop_size: 224
    # size_list = [270, 300, 330, 360]  # crop_size: 235
    # size_list = [348, 384, 420, 456]  # crop_size: 299
    # size_list = [352, 384, 416, 448]  # crop_size: 320
    short_edge = min(img.shape[:2])
    for i in size_list:
        img_resize = cv2.resize(img, (img.shape[1] * i / short_edge, img.shape[0] * i / short_edge))
        yy = int((img_resize.shape[0] - i) / 2)
        xx = int((img_resize.shape[1] - i) / 2)
        for j in xrange(3):
            left_center_right = img_resize[yy * j: yy * j + i, xx * j: xx * j + i]
            mirror = left_center_right[:, ::-1]
            crop_list.extend(over_sample(left_center_right))
            crop_list.extend(over_sample(mirror))
    return crop_list


def caffe_process(_input):
    _input = np.asarray(_input, dtype=np.float32)
    _input -= mean_value
    _input = _input.transpose(2, 0, 1)
    _input = _input.reshape((1,) + _input.shape)
    net.blobs['data'].reshape(*_input.shape)
    net.blobs['data'].data[...] = _input / raw_scale
    net.forward()
    _score = net.blobs[prob_layer].data[0]

    return _score


if __name__ == '__main__':
    eval_batch()
