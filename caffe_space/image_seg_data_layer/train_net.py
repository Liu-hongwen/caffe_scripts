import sys
# sys.setrecursionlimit(100000)

sys.path.append('/home/yanglu/workspace/deeplab-public-ver2-7752d9d6d676/python')

import caffe

weights = './ResNet-101-model.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(2)


solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

for _ in range(40000):
    solver.step(1)
