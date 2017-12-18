#!/usr/bin/env python3

import gzip
import requests
import numpy as np


class MNISTDataLoader(object):
    def load(self):
        return self.load_local()

    def load_local(self):
        with gzip.open('data/train-images-idx3-ubyte.gz') as f:
            training_images = self._parse_image_data(f.read())
        with gzip.open('data/train-labels-idx1-ubyte.gz') as f:
            training_labels = self._parse_label_data(f.read())
        with gzip.open('data/t10k-images-idx3-ubyte.gz') as f:
            test_images = self._parse_image_data(f.read())
        with gzip.open('data/t10k-labels-idx1-ubyte.gz') as f:
            test_labels = self._parse_label_data(f.read())
        return zip(training_images, training_labels), zip(test_images, test_labels)

    def load_remote(self):
        # home page: http://yann.lecun.com/exdb/mnist/
        r = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        training_images = self._parse_image_data(gzip.decompress(r.content))
        r = requests.get('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        training_labels = self._parse_label_data(gzip.decompress(r.content))
        r = requests.get('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        test_images = self._parse_image_data(gzip.decompress(r.content))
        r = requests.get('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
        test_labels = self._parse_label_data(gzip.decompress(r.content))
        return zip(training_images, training_labels), zip(test_images, test_labels)

    @staticmethod
    def _parse_image_data(content):
        p = 0
        # magic number
        magic = content[p:p+4]
        if magic != b'\x00\x00\x08\x03':
            exit("error magic number")
        p += 4

        # image count
        images = int.from_bytes(content[p:p+4], 'big')
        p += 4
        rows = int.from_bytes(content[p:p+4], 'big')
        p += 4
        columns = int.from_bytes(content[p:p+4], 'big')
        p += 4
        # pixels within an image
        pixels = rows * columns
        #
        X = [None] * images
        for i in range(images):
            X[i] = np.frombuffer(content[p:p+pixels], np.uint8)
            p += pixels
        return X

    @staticmethod
    def _parse_label_data(content):
        p = 0
        if content[p:p+4] != b'\x00\x00\x08\x01':
            exit('error magic number')
        p += 4
        #
        labels = int.from_bytes(content[p:p+4], 'big')
        p += 4
        return [MNISTDataLoader._one_hot_encode(k, 10) for k in np.frombuffer(content[p:p + labels], np.uint8)]

    @staticmethod
    def _one_hot_encode(k, n):
        encoded = np.zeros(n, np.uint8)
        encoded[k] = 1
        return encoded


if __name__ == '__main__':
    loader = MNISTDataLoader()
    loader.load()
