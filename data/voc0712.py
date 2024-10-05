"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
# VOC 的数据类别
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
# VOC 所在的目录
# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")

'''
为了提取 VOC 数据集中每张原图的 xml 文件的 bbox 坐标进行归一化，并将类别转化为字典格式，最后把数据组合起来
形状最后类似于 [[x_min, y_min, x_max, y_max, c], ...]
'''
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    
    将 VOC 的 annotation 中的 bbox 坐标转化为归一化的值；
    将类别转化为用索引来表示的字典形式；
    参数列表：
        class_to_ind: 类别的索引字典
        keep_difficult: 是否保留 difficult = 1 的物体
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        
        参数列表:
            target: xml 被读取的一个 ET.Element 对象
            width: 图片宽度
            height: 图片高度
        返回值:
            一个 list, list 中的每个元素是 [[bbox coords, class name]]
        """
        res = []
        for obj in target.iter('object'):
            # 判断目标是否 difficult
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            # 读取 xml 中所需的信息
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # xml 文件中 bbox 的表示
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # 归一化，x/w, y/h
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # 提取类别名称对应的 index
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """

    根据 VOCAnnotationTransform() 和 VOC 数据集的文件结构, 读取图片
    bbox 和 label, 构建 VOC 数据集的 DataLoader.

    VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    target_transform 传入上面的 VOCAnnotationTransform() 类
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # bbox 和 label
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        # 图片路径
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    # 可以自定义的函数
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # label 信息
        target = ET.parse(self._annopath % img_id).getroot()
        # 读取图片信息
        img = cv2.imread(self._imgpath % img_id)
        # 图片的长宽通道数
        height, width, channels = img.shape
        # 标签执行 VOCAnnotationTransform() 操作
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        # 数据(包括标签)是否需要执行 transform(数据增强)操作
        if self.transform is not None:
            target = np.array(target)
            # 执行了数据增强操作
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            # 把图片转化为 RGB
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            # 把 bbox 和 label 合并为 shape(N, 5)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        以 PIL 图像的方式返回下标为 index 的 PIL 格式原始图像
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        
        返回索引为 index 的图像的 xml 标注信息对象
        shape: [img_id, [(label, bbox coords), ...]]
        例子: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        
        以 Tensor 的形式返回索引为 index 的原始图像, 调用 unsqueeze_ 函数
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


'''
### 下面提供了一段代码可以测试上面的两个类的效果.


'''

Data = VOCDetection(VOC_ROOT)
data_loader = data.DataLoader(Data, batch_size=1,
                                    num_workers=0,
                                    shuffle=True,
                                    pin_memory=True)
print('the data length is: ', len(data_loader))

# 类别 to index
class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

# index to class, 转化为类别名称
ind_to_class = ind_to_class = {v:k for k, v in class_to_ind.items()}

# 加载数据
for datas in data_loader:
    img, target, h, w = datas
    img = img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    target = target[0].float()

    # 把 bbox 的坐标还原为原图的数值
    target[:,0] *= w.float()
    target[:,2] *= w.float()
    target[:,1] *= h.float()
    target[:,3] *= h.float()
    
    # 取整
    target = np.int0(target.numpy())
    # 画出图中类别名称
    for i in range(target.shape[0]):
        # 画矩形框
        img = cv2.rectangle(img, (target[i, 0], target[i, 1]), (target[i, 2], target[i, 3], (0, 0, 255), 2))
        img = cv2.putText(img, ind_to_class[target[i, 4]], (target[i, 0], target[i, 1] - 25))
    # 显示
    cv2.imshow('imgs', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break

