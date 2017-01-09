# --------------------------------------------------------
# augment_layer
# Copyright (c) 2016 DP Tech.
# Written by (Yang lu)
# --------------------------------------------------------
import sys

sys.path.append('/home/yanglu/workspace/deeplab_gcrf_insnorm_ohim/python')
import caffe
import numpy as np
import cv2


class OHIMLayer(caffe.Layer):
    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.ohim = params.get('ohim', True)
        self.top_k = params.get('top_k', 1)
        self.use_half_instance = params.get('use_half_instance', False)
        # self.instance_root = params.get('instance_root', 255)
        self.ignore_label = params.get('ignore_label', 255)
        self.ignore = params.get('ignore_instances', '0,255')
        self.ignore_instances = []
        for i in self.ignore.strip().split(','):
            self.ignore_instances.append(int(i))

        if len(top) != 1:
            raise Exception("Need to define only ont top: ohim_label.")
        if len(bottom) != 3:
            raise Exception("Need to define must three bottoms: predict, label, instance_label.")

    def forward(self, bottom, top):
        """

        :rtype: object
        bottom[0]: score map  1x151x41x41
        bottom[1]: label/8  1x1x41x41
        bottom[2]: instance/8  1x1x41x41
        top[0]: ohim_label  1x1x41x41
        """
        b_n, b_c, b_h, b_w = bottom[0].data[...].shape
        if self.ohim:
            # extending label
            extend_label = np.zeros(bottom[0].data[...].shape)
            for i in xrange((bottom[0]).num):
                for j in xrange(b_h):
                    for k in xrange(b_w):
                        cur_label = int(bottom[1].data[...][i][0][j][k])
                        if cur_label != 255:
                            extend_label[i][cur_label][j][k] = 1

            mini_batch = []
            for i in xrange((bottom[0]).num):
                new_label = np.ones((b_h, b_w), dtype=np.uint8) * self.ignore_label

                tmp_unique = np.unique(bottom[2].data[...][i])
                instance_label = [int(a) for a in tmp_unique if a not in self.ignore_instances]

                _loss = (np.array(bottom[0].data[...][i]) - np.array(extend_label[i])) * \
                        (np.array(bottom[0].data[...][i]) - np.array(extend_label[i]))
                _loss = np.sum(_loss, axis=0)  # 41x41

                ins_loss = np.zeros(len(instance_label))
                ins_cnt = np.zeros(len(instance_label))
                for h in xrange(b_h):
                    for w in xrange(b_w):
                        cur_label = int(bottom[2].data[...][i][0][h][w])
                        if cur_label in instance_label:
                            ins_loss[instance_label.index(cur_label)] += _loss[h][w]
                            ins_cnt[instance_label.index(cur_label)] += 1
                for j in xrange(len(instance_label)):
                    ins_loss[j] /= ins_cnt[j]
                # print instance_label, ins_loss

                hard_instance = []
                _idx = np.argsort(-np.asarray(ins_loss))
                if self.use_half_instance:
                    for j in xrange(1+len(instance_label)/2):
                        hard_instance.append(instance_label[_idx[j]])
                else:
                    for j in xrange(min(self.top_k, len(instance_label))):
                        hard_instance.append(instance_label[_idx[j]])
                # print hard_instance

                for h in xrange(b_h):
                    for w in xrange(b_w):
                        cur_label = int(bottom[2].data[...][i][0][h][w])
                        if cur_label in hard_instance:
                            new_label[h][w] = int(bottom[1].data[...][i][0][h][w])

                # cv2.imwrite('./label.png', cv2.resize(bottom[3].data[...][i][0], (321, 321)))
                cv2.imwrite('./label_shrink.png', cv2.resize(bottom[1].data[...][i][0], (321, 321)))
                cv2.imwrite('./ohim_label.png', cv2.resize(new_label, (321, 321)))
                mini_batch.append(np.asarray([new_label]))
            top[0].data[...] = mini_batch
        else:
            top[0].data[...] = bottom[1].data[...]

    def backward(self, top, propagate_down, bottom):
        """This layer does not need to backward propogate gradient"""
        pass

    def reshape(self, bottom, top):
        b1_shape = bottom[1].data[...].shape
        top[0].reshape(b1_shape[0], b1_shape[1], b1_shape[2], b1_shape[3])
