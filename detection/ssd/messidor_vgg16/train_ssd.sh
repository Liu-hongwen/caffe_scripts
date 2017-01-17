#!/usr/bin/env sh
set -e

/home/prmct/workspace/caffe-ssd-0103/build/tools/caffe train --gpu=3 \
    --solver=./proto/solver.prototxt \
    --weights=../models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
#     --snapshot=./VGG_glaucoma_SSD_800x800_iter_6892.solverstate
