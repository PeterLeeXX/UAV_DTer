import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random


class UAVDataset(Dataset):
    """
    单模态 UAV 数据集加载类，适配单模态 COCO 格式的数据集。
    支持处理有目标和无目标的样本，并应用常见的图像变换。
    """

    def __init__(self, annotation_file, image_dir, transform=None, image_size=800, multiscale=True):
        """
        Args:
            annotation_file (str): COCO 格式标注文件路径。
            image_dir (str): 图像文件目录路径。
            transform (callable, optional): 图像增强或预处理方法。
            image_size (int): 将图像的较小边调整为该大小，保持宽高比。
            multiscale (bool): 是否启用多尺度动态调整功能。
        """
        # 加载标注文件
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.multiscale = multiscale
        self.min_size = image_size - 3 * 32  # 最小尺寸
        self.max_size = image_size + 3 * 32  # 最大尺寸
        self.batch_count = 0  # 批次数计数
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # 创建 image_id 到 annotations 的映射
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回指定索引的图像及其对应的标注信息。

        Returns:
            img_path (str): 图像路径。
            img (Tensor): 图像张量。
            bb_targets (Tensor): 目标检测标签 [class, x, y, w, h]，格式与第一组代码一致。
        """
        # 获取图像信息
        image_info = self.images[idx]
        image_id = image_info['id']
        image_name = image_info['file_name']
        img_path = os.path.join(self.image_dir, image_name)

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        img = F.resize(img, size=self.image_size)

        # 获取对应的标注信息
        annotations = self.image_id_to_annotations.get(image_id, [])
        bb_targets = []

        for ann in annotations:
            bbox = ann['bbox']  # COCO 格式的边界框 [x, y, width, height]
            category_id = ann['category_id']
            if bbox:
                # 转换为 [class, x, y, w, h] 格式
                bb_targets.append([category_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        # 转换为张量
        bb_targets = torch.tensor(bb_targets, dtype=torch.float32) if bb_targets else torch.zeros((0, 5),
                                                                                                  dtype=torch.float32)

        # 应用图像变换
        if self.transform:
            img = self.transform(img)

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        """
        自定义 collate_fn，用于目标检测任务的批处理。
        - 支持动态多尺度调整。
        - 跳过无效样本。

        Args:
            batch (list): 单次加载的样本列表，每个样本是 (img_path, img, bb_targets)。

        Returns:
            paths (list): 图像路径列表。
            imgs (Tensor): 堆叠后的图像张量，形状为 [batch_size, 3, H, W]。
            bb_targets (Tensor): 合并后的目标检测标签，形状为 [num_targets, 6]。
                                每行格式为 [batch_idx, class, x, y, w, h]。
        """
        self.batch_count += 1

        # 移除无效样本
        batch = [data for data in batch if data is not None]

        # 如果批次为空，则返回空结果
        if len(batch) == 0:
            return [], torch.tensor([]), torch.tensor([])

        # 解包批次数据
        paths, imgs, bb_targets = list(zip(*batch))

        # 动态调整图像尺寸（每 10 个批次调整一次）
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # 调整图像大小
        imgs = torch.stack([F.resize(img, size=self.image_size) for img in imgs])

        # 处理目标检测标签
        for i, boxes in enumerate(bb_targets):
            if boxes.shape[0] > 0:
                # 在目标检测标签前添加 batch 索引
                boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)  # 合并所有样本的目标标签

        return paths, imgs, bb_targets


'''''''''
class UAVDataset(Dataset):
    """
    单模态 UAV 数据集加载类，适配单模态 COCO 格式的数据集。
    支持处理有目标和无目标的样本，并应用常见的图像变换。
    """

    def __init__(self, annotation_file, image_dir, transform=None, target_size=800):
        """
        Args:
            annotation_file (str): COCO 格式标注文件路径。
            image_dir (str): 图像文件目录路径。
            transform (callable, optional): 图像增强或预处理方法。
            target_size (int): 将图像的较小边调整为该大小，保持宽高比。
        """
        # 加载标注文件
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # 创建 image_id 到 annotations 的映射，避免重复标注
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            # 防止重复标注的累积
            if ann not in self.image_id_to_annotations[image_id]:
                self.image_id_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回指定索引的图像及其对应的标注信息。

        Returns:
            image (Tensor): 图像张量。
            target (dict): 标注信息，包含 'boxes' 和 'labels'。
        """
        # 获取图像信息
        image_info = self.images[idx]
        image_id = image_info['id']
        image_name = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_name)

        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # 原始尺寸 (width, height)

        # 动态调整图像大小，保持宽高比
        image = F.resize(image, size=self.target_size)

        # 获取对应的标注信息
        annotations = self.image_id_to_annotations.get(image_id, [])
        boxes = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']
            if bbox:  # 仅处理有目标的标注
                # bbox 格式：[x, y, width, height]
                boxes.append(bbox)
                labels.append(ann['category_id'])

        # 转换为张量
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # 构造目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(image_id, dtype=torch.int64),
            'original_size': torch.tensor(original_size, dtype=torch.int64),  # 原始尺寸
        }

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, target

    def collate_fn(batch):
        """
        自定义 collate_fn，用于 DataLoader 在目标检测任务中的批处理。
        """
        images = torch.stack([item[0] for item in batch], dim=0)  # 按照 batch 维度堆叠图像
        targets = [item[1] for item in batch]  # 保持每张图像对应的标注为列表形式
        return images, targets
'''''''''


class UAVMultimodalDataset(Dataset):
    """
    多模态 UAV 数据集加载类，适配多模态 COCO 格式的数据集。
    加载可见光 (visible) 和红外 (infrared) 图像对。
    """

    def __init__(self, annotation_file, visible_image_dir, infrared_image_dir, transform=None, target_size=800):
        """
        Args:
            annotation_file (str): COCO 格式标注文件路径（多模态）。
            visible_image_dir (str): 可见光图像目录。
            infrared_image_dir (str): 红外图像目录。
            transform (callable, optional): 图像增强或预处理方法。
            target_size (int): 将图像的较小边调整为该大小，保持宽高比。
        """
        # 加载标注文件
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.visible_image_dir = visible_image_dir
        self.infrared_image_dir = infrared_image_dir
        self.transform = transform
        self.target_size = target_size
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # 创建 image_id 到 annotations 的映射，避免重复标注
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回指定索引的数据，包括可见光和红外图像，以及对应的标注信息。
        Returns:
            visible_image (Tensor): 可见光图像张量。
            infrared_image (Tensor): 红外图像张量。
            target (dict): 标注信息，包括 'boxes_visible', 'boxes_infrared', 'labels' 等。
        """
        # 获取图像信息
        image_info = self.images[idx]
        image_id = image_info['id']

        # 加载可见光图像
        visible_image_name = image_info['file_name_visible']
        if os.path.isabs(visible_image_name):
            visible_image_path = os.path.normpath(visible_image_name)  # 已是完整路径
        else:
            visible_image_path = os.path.normpath(os.path.join(self.visible_image_dir, visible_image_name))
        visible_image = Image.open(visible_image_path).convert("RGB")

        # 加载红外图像
        infrared_image_name = image_info['file_name_infrared']
        if os.path.isabs(infrared_image_name):
            infrared_image_path = os.path.normpath(infrared_image_name)  # 已是完整路径
        else:
            infrared_image_path = os.path.normpath(os.path.join(self.infrared_image_dir, infrared_image_name))
        infrared_image = Image.open(infrared_image_path).convert("RGB")

        # 动态调整图像大小，保持宽高比
        visible_image = F.resize(visible_image, size=self.target_size)
        infrared_image = F.resize(infrared_image, size=self.target_size)

        # 获取对应的标注信息
        annotations = self.image_id_to_annotations.get(image_id, [])
        boxes_visible = []
        boxes_infrared = []
        labels = []

        for ann in annotations:
            bbox_visible = ann['bbox_visible']
            bbox_infrared = ann['bbox_infrared']
            if bbox_visible and bbox_infrared:  # 仅处理有目标的标注
                boxes_visible.append(bbox_visible)
                boxes_infrared.append(bbox_infrared)
                labels.append(ann['category_id'])

        # 转换为张量
        boxes_visible = torch.tensor(boxes_visible, dtype=torch.float32) if boxes_visible else torch.zeros((0, 4), dtype=torch.float32)
        boxes_infrared = torch.tensor(boxes_infrared, dtype=torch.float32) if boxes_infrared else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # 构造目标字典
        target = {
            'boxes_visible': boxes_visible,
            'boxes_infrared': boxes_infrared,
            'labels': labels,
            'image_id': torch.tensor(image_id, dtype=torch.int64)
        }

        # 应用图像变换
        if self.transform:
            visible_image = self.transform(visible_image)
            infrared_image = self.transform(infrared_image)

        return visible_image, infrared_image, target

    def collate_fn_multimodal(batch):
        """
        自定义 collate_fn，用于 DataLoader 在目标检测任务中的批处理。
        """
        visible_images = torch.stack([item[0] for item in batch], dim=0)
        infrared_images = torch.stack([item[1] for item in batch], dim=0)
        targets = [item[2] for item in batch]
        return visible_images, infrared_images, targets


'''''''''
# 测试代码
def test_uav_dataset():
    # 数据集路径
    annotation_file = './UAV_COCO_Dataset/annotations/instances_train_visible.json'  # 替换为你的标注文件路径
    image_dir = './UAV_COCO_Dataset/train/visible'  # 替换为你的图像文件目录路径

    # 定义图像变换
    transform = T.Compose([
        T.ToTensor(),  # 转换为张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 标准化
        T.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    ])

    # 创建数据集
    dataset = UAVDataset(annotation_file, image_dir, transform=transform, target_size=800)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=UAVDataset.collate_fn)

    # 遍历数据集进行测试
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Targets: {targets}")
        print()

        # 测试是否所有数据都能正常加载
        if batch_idx >= 2:  # 仅测试前 2 个批次
            break


# 测试代码
def test_uav_multimodal_dataset():
    # 数据集路径
    annotation_file = './UAV_COCO_Dataset_Multimodal/annotations/instances_train_multimodal.json'  # 替换为你的多模态标注文件路径
    visible_image_dir = './UAV_COCO_Dataset_Multimodal/train/visible'  # 替换为你的可见光图像文件目录路径
    infrared_image_dir = './UAV_COCO_Dataset_Multimodal/train/infrared'  # 替换为你的红外图像文件目录路径

    # 定义图像变换
    transform = T.Compose([
        T.ToTensor(),  # 转换为张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 标准化
        T.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    ])

    # 创建数据集
    dataset = UAVMultimodalDataset(annotation_file, visible_image_dir, infrared_image_dir, transform=transform, target_size=500)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=UAVMultimodalDataset.collate_fn_multimodal)

    # 遍历数据集进行测试
    for batch_idx, (visible_images, infrared_images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Visible Image batch shape: {visible_images.shape}")
        print(f"  Infrared Image batch shape: {infrared_images.shape}")
        print(f"  Targets: {targets}")
        print()

        # 测试是否所有数据都能正常加载
        if batch_idx >= 2:  # 仅测试前 2 个批次
            break


if __name__ == "__main__":
    test_uav_dataset()
    test_uav_multimodal_dataset()
'''''''''