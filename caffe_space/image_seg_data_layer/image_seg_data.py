import sys

sys.path.append('/home/yanglu/workspace/deeplab-public-ver2-7752d9d6d676/python')

import caffe

import cv2
import numpy as np


class ImageSegDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        print self.param_str
        params = eval(self.param_str)

        self.color_factor = np.array(params.get('color_factor', (0.90, 1.10)))
        self.contrast_factor = np.array(params.get('contrast_factor', (0.75, 1.25)))
        self.brightness_factor = np.array(params.get('brightness_factor', (0.75, 1.25)))
        self.mirror = params.get('mirror', True)
        self.gaussian_blur = params.get('gaussian_blur', False)
        self.scale_factor = np.array(params.get('scale_factor', (0.5, 2.0)))
        self.rotation_factor = np.array(params.get('rotation_factor', (-10, 10)))

        self.crop_size = int(params.get('crop_size', -1))
        self.mean = np.array(params.get('mean', (0.0, 0.0, 0.0)), dtype=np.float32)
        self.scale = float(params.get('scale', 1.0))

        self.root_dir = params['root_dir']
        self.source = params['source']
        self.batch_size = int(params.get('batch_size', 1))
        self.shuffle = params.get('shuffle', True)

        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")  # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        if len(self.color_factor) != 2:
            raise Exception("'color_factor' must have 2 values for factor range.")
        if len(self.contrast_factor) != 2:
            raise Exception("'contrast_factor' must have 2 values for factor range.")
        if len(self.brightness_factor) != 2:
            raise Exception("'brightness_factor' must have 2 values for factor range.")
        if len(self.mean) != 3:
            raise Exception("'mean' must have 3 values for B G R.")
        if len(self.scale_factor) != 2:
            raise Exception("'scale_factor' must have 2 values for factor range.")

        self.indices = open(self.source, 'r').read().splitlines()
        self.epoch_num = len(self.indices)
        self.idx = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        top[1].reshape(self.batch_size, 1, self.crop_size, self.crop_size)

    def forward(self, bottom, top):
        batch_img = []
        batch_label = []
        for _ in xrange(self.batch_size):
            _img = cv2.imread('{}{}'.format(self.root_dir, self.indices[self.idx].split(' ')[0]))
            _label = cv2.imread('{}{}'.format(self.root_dir, self.indices[self.idx].split(' ')[1]), 0)

            if _img.shape[:2] != _label.shape:
                raise Exception("Need to define two tops: data and label.")

            img, label = self.augmentation(_img, _label)
            batch_img.append(img.transpose((2, 0, 1)))
            batch_label.append([label])

            self.idx += 1
            if self.idx == self.epoch_num:
                self.idx = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
        batch_img = np.asarray(batch_img)
        batch_label = np.asarray(batch_label)

        # top[0].reshape(*batch_img.shape)
        # top[1].reshape(*batch_label.shape)

        top[0].data[...] = batch_img
        top[1].data[...] = batch_label

    def backward(self, top, propagate_down, bottom):
        pass

    def augmentation(self, img, label):
        ori_h, ori_w, ori_d = img.shape

        _color = 1.0
        _contrast = 1.0
        _brightness = 1.0

        if self.color_factor[0] != 0 and self.color_factor[1] != 0 and self.color_factor[0] < self.color_factor[1]:
            _color = np.random.randint(int(self.color_factor[0] * 100),
                                       int(self.color_factor[1] * 100)) / 100.0

        if self.contrast_factor[0] != 0 and self.contrast_factor[1] != 0 and self.contrast_factor[0] < \
                self.contrast_factor[1]:
            _contrast = np.random.randint(int(self.contrast_factor[0] * 100),
                                          int(self.contrast_factor[1] * 100)) / 100.0

        if self.brightness_factor[0] != 0 and self.brightness_factor[1] != 0 and self.brightness_factor[0] < \
                self.brightness_factor[1]:
            _brightness = np.random.randint(int(self.brightness_factor[0] * 100),
                                            int(self.brightness_factor[1] * 100)) / 100.0

        _HSV = np.dot(cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape((-1, 3)),
                      np.array([[_color, 0, 0], [0, _contrast, 0], [0, 0, _brightness]]))
        _HSV_H = np.where(_HSV < 255, _HSV, 255)
        img = cv2.cvtColor(np.uint8(_HSV_H.reshape((-1, img.shape[1], 3))), cv2.COLOR_HSV2BGR)

        if self.gaussian_blur:
            if not np.random.randint(0, 4):
                img = cv2.GaussianBlur(img, (3, 3), 0)

        img = np.asarray(img, dtype=np.float32)
        label = np.asarray(label, dtype=np.uint8)

        if self.mirror:
            if np.random.randint(0, 2):
                img = img[:, :: -1]
                label = label[:, :: -1]

        if self.scale_factor[0] != 0 and self.scale_factor[1] != 0 and self.scale_factor[0] < self.scale_factor[1]:
            _scale = np.random.randint(int(self.scale_factor[0] * 100),
                                       int(self.scale_factor[1] * 100)) / 100.0
            res_w = int(_scale * ori_w)
            res_h = int(_scale * ori_h)
            if min(res_w, res_h) <= self.crop_size:
                _ratio = float(self.crop_size) / min(res_w, res_h)
                res_w = int(_ratio * res_w) + np.random.randint(2, 5)
                res_h = int(_ratio * res_h) + np.random.randint(2, 5)
            img = cv2.resize(img, (res_w, res_h))
            label = cv2.resize(label, (res_w, res_h), interpolation=cv2.cv.CV_INTER_NN)
        else:
            res_w = int(1.0 * ori_w)
            res_h = int(1.0 * ori_h)
            if min(ori_w, ori_h) <= self.crop_size:
                _ratio = float(self.crop_size) / min(ori_w, ori_h)
                res_w = int(_ratio * ori_w) + np.random.randint(2, 5)
                res_h = int(_ratio * ori_h) + np.random.randint(2, 5)
            img = cv2.resize(img, (res_w, res_h))
            label = cv2.resize(label, (res_w, res_h), interpolation=cv2.cv.CV_INTER_NN)

        if self.rotation_factor[0] != 0 and self.rotation_factor[1] != 0 and self.rotation_factor[0] < \
                self.rotation_factor[1]:
            if np.random.randint(0, 2):
                _rotation = np.random.randint(int(self.rotation_factor[0] * 100),
                                              int(self.rotation_factor[1] * 100)) / 100.0
                tmp_h, tmp_w, tmp_d = img.shape
                rotate_mat = cv2.getRotationMatrix2D((tmp_w / 2, tmp_h / 2), _rotation, 1)
                img = cv2.warpAffine(img, rotate_mat, (tmp_w, tmp_h))
                label = cv2.warpAffine(label, rotate_mat, (tmp_w, tmp_h), flags=cv2.cv.CV_INTER_NN, borderValue=255)

        aug_h, aug_w, aug_d = img.shape
        if self.crop_size > 0:
            offset_h = np.random.randint(0, aug_h - self.crop_size)
            offset_w = np.random.randint(0, aug_w - self.crop_size)
            img = img[offset_h:offset_h + self.crop_size, offset_w:offset_w + self.crop_size, :]
            label = label[offset_h:offset_h + self.crop_size, offset_w:offset_w + self.crop_size]

        img -= self.mean
        img *= self.scale

        return img, label
