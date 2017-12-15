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
            self.conv_decode1_bn = L.BatchNormalization(64, initial_beta=0.001)
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

    def __call__(self, x):
        """
        Compute as image-wise score from a batch of images
        :param x: chainer.Variable: A variable with 40 image array.
        :return: chainer.Variable: An image-wise score. Its channel size is "self.n_class".
        """
        p1 = F.MaxPooling2D(2, 2)
        p2 = F.MaxPooling2D(2, 2)
        p3 = F.MaxPooling2D(2, 2)
        p4 = F.MaxPooling2D(2, 2)
        h = F.local_response_normalization(x, 5, 1, 1e-4 / 5.0, 0.75)
        h = _pool_without_cudnn(p1, F.relu(self.conv1_bn(self.conv1(h))))
        h = _pool_without_cudnn(p2, F.relu(self.conv2_bn(self.conv2(h))))
        h = _pool_without_cudnn(p3, F.relu(self.conv3_bn(self.conv3(h))))
        h = _pool_without_cudnn(p4, F.relu(self.conv4_bn(self.conv4(h))))
        h = self._upsampling_2d(h, p4)
        h = self.conv_decode4_bn(self.conv_decode4(h))
        h = self._upsampling_2d(h, p3)
        h = self.conv_decode3_bn(self.conv_decode3(h))
        h = self._upsampling_2d(h, p2)
        h = self.conv_decode2_bn(self.conv_decode2(h))
        h = self._upsampling_2d(h, p1)
        h = self.conv_decode1_bn(self.conv_decode1(h))
        score = self.conv_classifier(h)
        return score

    def predict(self, imgs):
        """
        Conduct semantic segmentations from images.

        :param imgs: iterable of numpy.ndarray:
                    Arrays holding images.
                    All images are CHW and RGB format
                    and the range of their values are :math:"[0, 255]".

        :return: list of numpy.ndarray:
                    List of integer lables predicted from
                    each image input list.
        """
        labels = list()
        for img in imgs:
            C, H, W = img.shape
            with chainer.using_config("train", False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.ndarray(img[np.newaxis]))
                score = self.__call__(x)[0].data
            score = chainer.cuda.to_cpu(score)
            if score.shape != (C, W, H):
                dtype = score.dtype
                score = resize(score, (H, W)).astype(dtype)

            label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels

