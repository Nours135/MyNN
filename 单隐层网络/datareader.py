
# 这个数据集的存储方式确实空间效率达到了极致，但是对没多少编程IO基础的人来说很不友好
# 具体数据结构参照 http://yann.lecun.com/exdb/mnist/
from torch import tensor
from random import shuffle

class MNISTReader():
    train_img_path = "train-images.idx3-ubyte"
    train_label_path = 'train-labels.idx1-ubyte'
    verify_img_path = 't10k-images.idx3-ubyte'
    verify_label_path = 't10k-labels.idx1-ubyte'

    label_file_metadata_bytes = 8   #前面元数据的大小，详情参见网站
    img_file_metadata_bytes = 16

    def __init__(self, mode=1):
        # mode 1是训练，mode 2是验证，读取不同的数据
        if mode == 1:
            self.img_f = open(self.train_img_path, 'rb')
            self.label_f = open(self.train_label_path, 'rb')
        elif mode == 2:
            self.img_f = open(self.verify_img_path, 'rb')
            self.label_f = open(self.verify_label_path, 'rb')
        else:
            pass

        # 读取文件头存储的metadata
        self.img_bytes = self.img_f.read()
        self.label_bytes = self.label_f.read()
        # 读取图片文件
        magic_number = int.from_bytes(self.img_bytes[:4], 'big', signed=True)
        if magic_number != 2051:
            raise Exception('读取出现问题，图片的magic number出错')
        self.total_img = int.from_bytes(self.img_bytes[4:8], 'big', signed=True)
        if ((len(self.img_bytes)-self.img_file_metadata_bytes)/784 - self.total_img) > 0.5:
            raise Exception('读取出现问题，图片bytes和图片数量对不上')

        # 读取label文件
        magic_number2 = int.from_bytes(self.label_bytes[:4], 'big', signed=True)
        if magic_number2 != 2049:
            raise Exception('读取出现问题，标签的magic number出错')
        total_img = int.from_bytes(self.label_bytes[4:8], 'big', signed=True)
        if total_img != self.total_img:
            raise Exception('读取出现错误，标签文件和图片文件记录的图片数不一致')
        if ((len(self.label_bytes)-self.label_file_metadata_bytes) - self.total_img) > 0.5:
            raise Exception('读取出现问题，标签文件bytes和图片数量不一致')

        self.img_f.close()
        self.label_f.close()

    def get_pic_count(self):
        # 返回一下图片总数，一个api
        return self.total_img

    def get_pic(self, id):
        if id == 0:
            raise Exception('id从1开始，大傻逼不细心吧！')
        # id从1开始数
        # 一次读取一个，暂时设计成这样，没必要，也懒得优化读取速度了，反正性能主要消耗在训练上，而不是IO
        self.cur_pic = [] # 暂时存储为784长度的列表吧，后续确认像素顺序再说
        self.cur_label = 0
        img_start = self.img_file_metadata_bytes + (id-1) * 784
        label_start = self.label_file_metadata_bytes + (id-1)

        for i in range(784):
            # 直接在最初的读取的时候就将数据全部归一化
            self.cur_pic.append((int.from_bytes(self.img_bytes[img_start: img_start+1], 'big', signed=False)-128)/128)
            img_start += 1
        self.cur_label = int.from_bytes(self.label_bytes[label_start: label_start+1], 'big', signed=False)
        #print(len(self.cur_pic))
        #print(self.cur_pic)
        #print(self.cur_label)

        return [self.cur_pic, self.cur_label]

    def show_img(self):
        # 因为前面将数据归一化了，所以需要*255，恢复它，不保证没bug，因为反正这个函数不重要
        #from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt

        imdata = np.array(self.cur_pic)
        imdata = imdata.reshape((28, 28))
        #print(imdata)
        fig = plt.figure()
        plt.imshow(imdata)
        plt.show()

    def dataIter(self, BatchSize):
        indices = list(range(1, 1+self.total_img))
        shuffle(indices)
        for i in range(0, self.total_img, BatchSize):
            batch_indices = indices[i: min(i+BatchSize, self.total_img)]
            features = []
            labels = []
            for id in batch_indices:
                re = self.get_pic(id)
                if re[1] == 96:
                    print(id)
                    print(re[0])
                    raise IndexError
                features.append(re[0])
                labels.append(re[1])
            yield tensor(features), tensor(labels)



if __name__ == '__main__':
    test = MNISTReader(2)
    for p in range(9990, 10000):
        test.get_pic(p)
        test.show_img()
        input('aa: ')




