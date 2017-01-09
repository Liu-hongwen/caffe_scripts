# -*- coding:utf-8 -*-
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('/Users/yanglu/workspace/caffe-master-0503/python')

import caffe
import torchfile

torch_model = torchfile.load('/Users/yanglu/data_file/Y2_es.t7')
models = torch_model['model'].modules
net = caffe.Net('./caffe_style.prototxt', caffe.TEST)
for index in xrange(3):
    net.params['conv' + str(index + 1)][0].data[...] = models[3 * index + 1].weight
    net.params['conv' + str(index + 1)][1].data[...] = models[3 * index + 1].bias
    # net.params['bn_conv'+str(index+1)][0].data[...] = models[3*index+1].bn.running_mean
    print net.params['bn_conv1'][2].shape
    # net.params['bn_conv'+str(index+1)][1].data[...] = models[3*index+1].bn.running_var
    net.params['scale_conv' + str(index + 1)][0].data[...] = models[3 * index + 2].weight
    net.params['scale_conv' + str(index + 1)][1].data[...] = models[3 * index + 2].bias
    if index < 2:
        net.params['dconv' + str(index + 1)][0].data[...] = models[3 * index + 15].weight
        net.params['dconv' + str(index + 1)][1].data[...] = models[3 * index + 15].bias
        #       net.params['bn_dconv'+str(index+1)][0].data[...] = models[3*index+16].bn.running_mean
        #        net.params['bn_dconv'+str(index+1)][1].data[...] = models[3*index+16].bn.running_var

        net.params['scale_dconv' + str(index + 1)][0].data[...] = models[3 * index + 16].weight
        net.params['scale_dconv' + str(index + 1)][1].data[...] = models[3 * index + 16].bias
    else:
        net.params['dconv' + str(index + 1)][0].data[...] = models[21].weight
for index in xrange(5):
    res_models = models[10 + index].modules[0].modules[0].modules
    net.params['r' + str(index + 1) + '_conv1'][0].data[...] = res_models[0].weight
    net.params['r' + str(index + 1) + '_conv1'][1].data[...] = res_models[0].bias
    #    net.params['r'+str(index +1)+'_bn_conv1'][0].data[...] = res_models[1].bn.running_mean
    #    net.params['r'+str(index +1)+'_bn_conv1'][1].data[...] = res_models[1].bn.running_var
    net.params['r' + str(index + 1) + '_scale_conv1'][0].data[...] = res_models[1].weight
    net.params['r' + str(index + 1) + '_scale_conv1'][1].data[...] = res_models[1].bias
    net.params['r' + str(index + 1) + '_conv2'][0].data[...] = res_models[3].weight
    net.params['r' + str(index + 1) + '_conv2'][1].data[...] = res_models[3].bias
    # net.params['r'+str(index +1)+'_bn_conv2'][0].data[...] = res_models[4].bn.running_mean
    # net.params['r'+str(index +1)+'_bn_conv2'][1].data[...] = res_models[4].bn.running_var
    net.params['r' + str(index + 1) + '_scale_conv2'][0].data[...] = res_models[4].weight
    net.params['r' + str(index + 1) + '_scale_conv2'][1].data[...] = res_models[4].bias
net.save('111.caffemodel')
