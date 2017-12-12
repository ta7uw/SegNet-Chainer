import argparse
import numpy as np
from collections import defaultdict
import chainer

from chainer.dataset import concat_examples
from chainercv.datasets import camvid_label_colors, CamVidDataset
from chainercv.utils import apply_prediction_to_iterator

from segnet import SegNetBasic


def calc_bn_statistics(model, batchsize):
    train = CamVidDataset(split="train")
    it = chainer.iterators.SerialIterator(
        train, batchsize, repeat=False, shuffle=False
    )
    bn_avg_mean = defaultdict(np.float32)
    bn_avg_var = defaultdict(np.float32)

    n_iter = 0
    for batch in it:
        imgs, _ = concat_examples(batch)
        model(model.xp.array(imgs))
        for name, link in model.namedlinks():
            if name.endwith("_bn"):
                bn_avg_mean[name] += link.avg_mean
                bn_avg_var[name] += link.avg_var
        n_iter += 1

    for name, link in model.namedlinks():
        if name.endwith("_bn"):
            link.avg_mean = bn_avg_mean[name] / n_iter
            link.avg_var = bn_avg_var[name]/ n_iter

    return model
