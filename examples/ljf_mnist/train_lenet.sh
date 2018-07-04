#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/ljf_mnist/lenet_solver.prototxt $@
