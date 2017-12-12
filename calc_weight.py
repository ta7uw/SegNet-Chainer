import numpy as np

from chainercv.datasets import CamVidDataset


def main():
    n_class = 11
    dataset = CamVidDataset(split="train")

    n_cls_pixels = np.zeros((n_class,))
    n_img_pizels = np.zeros((n_class,))

    for img, label in dataset:
        for cls_i in np.unique(label):
            if cls_i == -1:
                continue
            n_cls_pixels[cls_i] += np.sum(label == cls_i)
            n_img_pizels[cls_i] += label.size

    freq = n_cls_pixels / n_img_pizels
    median_freq = np.median(freq)
    np.save("class_weight", median_freq / freq)

if __name__ == '__main__':
    main()