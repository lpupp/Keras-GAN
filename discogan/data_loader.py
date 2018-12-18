import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, img_res=(128, 128)):
        self.dataset_e2s = 'edges2shoes'
        self.dataset_e2h = 'edges2handbags'
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_e2s = glob('./datasets/%s/%s/*' % (self.dataset_e2s, data_type))
        path_e2h = glob('./datasets/%s/%s/*' % (self.dataset_e2h, data_type))

        batch_e2s = np.random.choice(path_e2s, size=batch_size)
        batch_e2h = np.random.choice(path_e2h, size=batch_size)

        imgs_s, imgs_h = [], []
        for img_e2s, img_e2h in zip(batch_e2s, batch_e2h):
            img_s = self.imread(img_e2s)
            img_h = self.imread(img_e2h)

            img_s = scipy.misc.imresize(img_s, self.img_res)
            img_h = scipy.misc.imresize(img_h, self.img_res)

            if not is_testing and np.random.random() > 0.5:
                    img_s = np.fliplr(img_s)
                    img_h = np.fliplr(img_h)

            imgs_s.append(img_s)
            imgs_h.append(img_h)

        imgs_s = np.array(imgs_s)/127.5 - 1.
        imgs_h = np.array(imgs_h)/127.5 - 1.

        return imgs_s, imgs_h

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_e2s = glob('./datasets/%s/%s/*' % (self.dataset_e2s, data_type))
        path_e2h = glob('./datasets/%s/%s/*' % (self.dataset_e2h, data_type))

        self.n_batches_s = int(len(path_e2s) / batch_size)
        self.n_batches_h = int(len(path_e2h) / batch_size)
        self.n_batches = max(self.n_batches_s, self.n_batches_h)

        j = 0
        for i in range(self.n_batches-1):
            batch_h = path_e2h[i*batch_size:(i+1)*batch_size]
            imgs_s, imgs_h = [], []
            for img in batch_h:
                img_h = self.imread(img)
                img_h = scipy.misc.imresize(img_h, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_h = np.fliplr(img_h)

                imgs_h.append(img_h)

            try:
                batch_s = path_e2s[j*batch_size:(j+1)*batch_size]
            except IndexError:
                j = 0
                batch_s = path_e2s[j*batch_size:(j+1)*batch_size]

            for img in batch_s:
                img_s = self.imread(img)
                img_s = scipy.misc.imresize(img_s, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_s = np.fliplr(img_s)

                imgs_s.append(img_s)

            j += 1
            imgs_h = np.array(imgs_h)/127.5 - 1.
            imgs_s = np.array(imgs_s)/127.5 - 1.

            yield imgs_s, imgs_h

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
