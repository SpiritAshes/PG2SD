import numpy as np
from PIL import Image

from .dataset import Dataset, CatDataset
from utils.transforms import instanciate_transformation
from utils.transforms_tools import persp_apply


class PairDataset (Dataset):
    """ A dataset that serves image pairs with ground - truth pixel correspondences.
    一个用于提供带有真实像素对应关系的图像对的数据集类
    """
    def __init__(self):
        # 调用父类 Dataset 的构造函数
        Dataset.__init__(self)
        # 初始化图像对的数量为 0
        self.npairs = 0

    def get_filename(self, img_idx, root=None):
        # 如果 img_idx 是一个图像对的索引
        if is_pair(img_idx): 
            # 返回一个包含两个图像文件名的元组
            return tuple(Dataset.get_filename(self, i, root) for i in img_idx)
        # 否则返回单个图像的文件名
        return Dataset.get_filename(self, img_idx, root)

    def get_image(self, img_idx):
        # 如果 img_idx 是一个图像对的索引
        if is_pair(img_idx): 
            # 返回一个包含两个图像的元组
            return tuple(Dataset.get_image(self, i) for i in img_idx)
        # 否则返回单个图像
        return Dataset.get_image(self, img_idx)

    def get_corres_filename(self, pair_idx):
        # 此方法需要在子类中实现，这里抛出未实现错误
        raise NotImplementedError()

    def get_homography_filename(self, pair_idx):
        # 此方法需要在子类中实现，这里抛出未实现错误
        raise NotImplementedError()

    def get_flow_filename(self, pair_idx):
        # 此方法需要在子类中实现，这里抛出未实现错误
        raise NotImplementedError()

    def get_mask_filename(self, pair_idx):
        # 此方法需要在子类中实现，这里抛出未实现错误
        raise NotImplementedError()

    def get_pair(self, idx, output=()):
        """ 
        返回 (img1, img2, `元数据`)
        `元数据` 是一个字典，可能包含以下内容：
            flow: 光流
            aflow: 绝对光流
            corres: 二维到二维的对应关系列表
            mask: 光流有效性的布尔图像（在第一张图像中）
        """
        # 此方法需要在子类中实现，这里抛出未实现错误
        raise NotImplementedError()

    def get_paired_images(self):
        # 用于存储所有图像对中涉及的图像文件名的集合
        fns = set()
        # 遍历所有图像对
        for i in range(self.npairs):
            # 获取当前图像对的两个图像索引
            a, b = self.image_pairs[i]
            # 将第一个图像的文件名添加到集合中
            fns.add(self.get_filename(a))
            # 将第二个图像的文件名添加到集合中
            fns.add(self.get_filename(b))
        return fns

    def __len__(self):
        # 数据集的长度定义为图像对的数量
        return self.npairs 

    def __repr__(self):
        # 生成一个用于描述数据集的字符串
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images,' % self.nimg
        res += ' %d image pairs' % self.npairs
        res += '\n  root: %s...\n' % self.root
        return res

    @staticmethod
    def _flow2png(flow, path):
        """
        将光流数据保存为 PNG 文件
        :param flow: 光流数据
        :param path: 保存的文件路径
        :return: 缩放后的光流数据
        """
        # 对光流数据进行缩放和四舍五入，并限制在 -2**15 到 2**15 - 1 范围内
        flow = np.clip(np.around(16*flow), -2**15, 2**15 - 1)
        # 将光流数据转换为 uint8 类型
        bytes = np.int16(flow).view(np.uint8)
        # 将数据保存为 PNG 图像
        Image.fromarray(bytes).save(path)
        # 返回缩放后的光流数据
        return flow / 16

    @staticmethod
    def _png2flow(path):
        """
        从 PNG 文件中读取光流数据
        :param path: PNG 文件的路径
        :return: 读取的光流数据
        """
        try:
            # 打开 PNG 文件并将其转换为 NumPy 数组
            flow = np.asarray(Image.open(path)).view(np.int16)
            # 将数据转换为 float32 类型并进行缩放
            return np.float32(flow) / 16
        except:
            # 如果读取失败，抛出 IO 错误
            raise IOError("Error loading flow for %s" % path)


class StillPairDataset (PairDataset):
    """
    两个图像对缩放，计算光流
    添加了 get_pair(i) 函数，该函数返回简单的图像对 (img1, img2)，其中 img1 == img2 == get_image(i)
    """
    def get_pair(self, pair_idx, output=()):
        # 如果 output 是字符串，将其拆分为列表
        if isinstance(output, str): output = output.split()
        # 获取图像对中的两个图像
        img1, img2 = map(self.get_image, self.image_pairs[pair_idx])

        # 获取第一张图像的宽度和高度
        W, H = img1.size
        # 计算第二张图像相对于第一张图像的宽度缩放比例
        sx = img2.size[0] / float(W)
        # 计算第二张图像相对于第一张图像的高度缩放比例
        sy = img2.size[1] / float(H)

        # 用于存储元数据的字典
        meta = {}
        # 如果需要输出光流或绝对光流
        if 'aflow' in output or 'flow' in output:
            # 生成一个网格，用于表示图像中的每个像素坐标
            mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1, 2, 0).astype(np.float32)
            # 计算绝对光流
            meta['aflow'] = mgrid * (sx, sy)
            # 计算光流
            meta['flow'] = meta['aflow'] - mgrid

        # 如果需要输出掩码
        if 'mask' in output:
            # 生成一个全为 1 的掩码矩阵
            meta['mask'] = np.ones((H, W), np.uint8)

        # 如果需要输出单应性矩阵
        if 'homography' in output:
            # 生成一个对角矩阵作为单应性矩阵
            meta['homography'] = np.diag(np.float32([sx, sy, 1]))

        return img1, img2, meta


class SyntheticPairDataset (PairDataset):
    """
    一个用于生成合成图像对的数据集类。
    给定一个普通图像数据集，使用随机单应性矩阵和噪声构建图像对
    """
    def __init__(self, dataset, scale='', distort=''):
        # 将传入的数据集附加到当前对象
        self.attach_dataset(dataset)
        # 实例化缩放扭曲变换
        self.distort = instanciate_transformation(distort)
        self.scale = instanciate_transformation(scale)

    def attach_dataset(self, dataset):
        # 确保传入的数据集是 Dataset 类的实例，但不是 PairDataset 类的实例
        assert isinstance(dataset, Dataset) and not isinstance(dataset, PairDataset)

        self.dataset = dataset
        self.npairs = dataset.nimg
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None

    def make_pair(self, img):
        # 简单地返回相同的图像对
        return img, img

    def get_pair(self, i, output=('aflow')):
        """ 
        此函数对一张原始图像应用一系列随机变换，以形成具有真实标签的合成图像对
        """
        # 如果 output 是字符串，将其拆分为列表
        if isinstance(output, str): 
            output = output.split()

        # 从传入的数据集中获取第 i 张图像
        original_img = self.dataset.get_image(i)

        # 对原始图像应用缩放变换
        scaled_image = self.scale(original_img)
        # 生成图像对
        scaled_image, scaled_image2 = self.make_pair(scaled_image)
        # 对第二张缩放后的图像应用扭曲变换
        scaled_and_distorted_image = self.distort(
            dict(img=scaled_image2, persp=(1, 0, 0, 0, 1, 0, 0, 0)))
        # 获取缩放后图像的宽度和高度
        W, H = scaled_image.size
        # 获取扭曲变换的参数
        trf = scaled_and_distorted_image['persp']

        # 用于存储元数据的字典
        meta = dict()
        # 如果需要输出光流或绝对光流
        if 'aflow' in output or 'flow' in output:
            # 生成一个网格，用于表示图像中的每个像素坐标
            xy = np.mgrid[0:H, 0:W][::-1].reshape(2, H * W).T
            # 计算绝对光流
            aflow = np.float32(persp_apply(trf, xy).reshape(H, W, 2))
            # 计算光流
            meta['flow'] = aflow - xy.reshape(H, W, 2)
            meta['aflow'] = aflow

        # 如果需要输出单应性矩阵
        if 'homography' in output:
            # 生成单应性矩阵
            meta['homography'] = np.float32(trf+(1,)).reshape(3, 3)

        return scaled_image, scaled_and_distorted_image['img'], meta

    def __repr__(self):
        # 生成一个用于描述数据集的字符串
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images and pairs' % self.npairs
        res += '\n  root: %s...' % self.dataset.root
        res += '\n  Scale: %s' % (repr(self.scale).replace('\n', ''))
        res += '\n  Distort: %s' % (repr(self.distort).replace('\n', ''))
        return res + '\n'


class TransformedPairs (PairDataset):
    """
    对已有图像对进行自动数据增强的数据集类。
    给定一个图像对数据集，使用随机变换（如单应性矩阵和噪声）生成合成的抖动图像对
    """
    def __init__(self, dataset, trf=''):
        # 将传入的数据集附加到当前对象
        self.attach_dataset(dataset)
        # 实例化变换操作
        self.trf = instanciate_transformation(trf)

    def attach_dataset(self, dataset):
        # 确保传入的数据集是 PairDataset 类的实例
        assert isinstance(dataset, PairDataset)

        self.dataset = dataset
        self.nimg = dataset.nimg
        self.npairs = dataset.npairs
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None

    def get_pair(self, i, output=''):
        """ 
        此函数对一张原始图像应用一系列随机变换，以形成具有真实标签的合成图像对
        """
        # 从传入的数据集中获取第 i 个图像对及其元数据
        img_a, img_b_, metadata = self.dataset.get_pair(i, output)

        # 对第二张图像应用变换操作
        img_b = self.trf({'img': img_b_, 'persp': (1, 0, 0, 0, 1, 0, 0, 0)})
        # 获取变换的参数
        trf = img_b['persp']

        # 如果元数据中包含光流或绝对光流
        if 'aflow' in metadata or 'flow' in metadata:
            # 获取绝对光流
            aflow = metadata['aflow']
            # 对绝对光流应用变换
            aflow[:] = persp_apply(trf, aflow.reshape(-1, 2)).reshape(aflow.shape)
            # 获取第一张图像的宽度和高度
            W, H = img_a.size
            # 获取光流
            flow = metadata['flow']
            # 生成一个网格，用于表示图像中的每个像素坐标
            mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1, 2, 0).astype(np.float32)
            # 重新计算光流
            flow[:] = aflow - mgrid

        # 如果元数据中包含对应关系
        if 'corres' in metadata:
            # 获取对应关系
            corres = metadata['corres']
            # 对对应关系的第二列应用变换
            corres[:, 1] = persp_apply(trf, corres[:, 1])

        # 如果元数据中包含单应性矩阵
        if 'homography' in metadata:
            # p_b = homography * p_a
            # 将变换参数转换为 3x3 矩阵
            trf_ = np.float32(trf+(1,)).reshape(3, 3)
            # 更新单应性矩阵
            metadata['homography'] = np.float32(trf_ @ metadata['homography'])

        return img_a, img_b['img'], metadata

    def __repr__(self):
        # 生成一个用于描述数据集的字符串
        res =  'Transformed Pairs from %s\n' % type(self.dataset).__name__
        res += '  %d images and pairs' % self.npairs
        res += '\n  root: %s...' % self.dataset.root
        res += '\n  transform: %s' % (repr(self.trf).replace('\n', ''))
        return res + '\n'


class CatPairDataset (CatDataset):
    ''' Concatenation of several pair datasets.
    多个图像对数据集的拼接类
    '''
    def __init__(self, *datasets):
        # 调用父类 CatDataset 的构造函数
        CatDataset.__init__(self, *datasets)
        # 用于存储每个数据集的图像对偏移量的列表，初始值为 0
        pair_offsets = [0]
        # 遍历所有传入的数据集
        for db in datasets:
            # 将当前数据集的图像对数量添加到偏移量列表中
            pair_offsets.append(db.npairs)
        # 计算偏移量的累积和
        self.pair_offsets = np.cumsum(pair_offsets)
        # 总的图像对数量等于最后一个偏移量
        self.npairs = self.pair_offsets[-1]

    def __len__(self):
        # 数据集的长度定义为总的图像对数量
        return self.npairs

    def __repr__(self):
        # 生成一个用于描述数据集的字符串
        fmt_str = "CatPairDataset("
        for db in self.datasets:
            fmt_str += str(db).replace("\n", " ") + ', '
        return fmt_str[:-2] + ')'

    def pair_which(self, i):
        # 查找索引 i 所在的数据集位置
        pos = np.searchsorted(self.pair_offsets, i, side='right') - 1
        # 确保索引 i 小于总的图像对数量
        assert pos < self.npairs, 'Bad pair index %d >= %d' % (i, self.npairs)
        # 返回数据集位置和在该数据集中的相对索引
        return pos, i - self.pair_offsets[pos]

    def pair_call(self, func, i, *args, **kwargs):
        # 确定索引 i 所在的数据集和相对索引
        b, j = self.pair_which(i)
        # 调用对应数据集的指定方法
        return getattr(self.datasets[b], func)(j, *args, **kwargs)

    def get_pair(self, i, output=()):
        # 确定索引 i 所在的数据集和相对索引
        b, i = self.pair_which(i)
        # 调用对应数据集的 get_pair 方法
        return self.datasets[b].get_pair(i, output)

    def get_flow_filename(self, pair_idx, *args, **kwargs):
        # 调用对应数据集的 get_flow_filename 方法
        return self.pair_call('get_flow_filename', pair_idx, *args, **kwargs)

    def get_mask_filename(self, pair_idx, *args, **kwargs):
        # 调用对应数据集的 get_mask_filename 方法
        return self.pair_call('get_mask_filename', pair_idx, *args, **kwargs)

    def get_corres_filename(self, pair_idx, *args, **kwargs):
        # 调用对应数据集的 get_corres_filename 方法
        return self.pair_call('get_corres_filename', pair_idx, *args, **kwargs)


def is_pair(x):
    """
    判断输入是否为一个图像对
    :param x: 输入
    :return: 如果是图像对返回 True，否则返回 False
    """
    # 如果 x 是长度为 2 的元组或列表
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return True
    # 如果 x 是一维且长度为 2 的 NumPy 数组
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False