import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.transforms import resize
from chainercv.utils import download_model


def _pool_without_cudnn(p, x):
    with chainer.using_config("use_cudnn", "never"):
        return p.apply((x,))[0]


class SegNetBasic(chainer.Chain):
    """
    SegNet Basic for semantic segmentation.
    """

    _models = {
        "camvid": {
            "n_class": 11,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.2/segnet_camvid_2017_05_28.npz'
        }
    }

    def __init__(self, n_class=None, pretrained_model=None, initial_w=None):
        if n_class is None:
            if pretrained_model not in self._models:
                raise ValueError(
                    'The n_class needs to be supplied as an argument.'
                )
            n_class = self._models[pretrained_model]["n_class"]

        if initial_w is None:
            initial_w = chainer.initializers.HeNormal()

        super(SegNetBasic, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv1_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv2 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv2_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv3 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv3_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv4 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv4_bn = L.BatchNormalization(64, initial_beta=0.001)

            self.conv_decode4 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv_decode4_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv_decode3 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv_decode3_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv_decode2 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv_decode2_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv_decode1 = L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=initial_w)
            self.conv_decode_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv_classifier = L.Convolution2D(64, n_class, 1, 1, 0, initialW=initial_w)

        self.n_class = n_class

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]["url"])
            chainer.serializers.load_npz(path, self)

        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def _upsampling_2d(self, x, pool):
        if x.shape != pool.indexes.shape:
            min_h = min(x.shape[2], pool.indexes.shape[2])
            min_w = min(x.shape[3], pool.indexes.shape[3])
            x = x[:, :, : min_h, :min_w]
            pool.indexes = pool.indexes[:, :, :min_h, :min_w]
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(
            x, pool.indexes, ksize=(pool.kh, pool.kw),
            stride=(pool.sy, pool.sx), pad=(pool.ph, pool.pw), outsize=outsize
        )




