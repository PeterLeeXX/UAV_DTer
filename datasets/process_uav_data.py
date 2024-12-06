import os
import cv2
import json
from tqdm.autonotebook import tqdm


def decode_and_sample(video_path, output_dir, prefix, modality, interval=10, global_frame_id=1):
    """
    解码视频并抽取帧，保存为图像文件，同时确保同一模态内的图片具有全局唯一 ID，
    不同模态间的图片具有一一对应的 ID。

    Args:
        video_path (str): 视频路径。
        output_dir (str): 保存帧的文件夹路径。
        prefix (str): 文件名前缀（视频组名）。
        modality (str): 模态类型（visible 或 infrared）。
        interval (int): 每隔多少帧抽取一帧。
        global_frame_id (int): 当前模态的全局帧计数器。

    Returns:
        int: 更新后的全局帧计数器。
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # 当前视频的帧计数
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        frame_count += 1  # 始终自增帧计数器
        if not ret:  # 视频结束
            break

        # 每 interval 帧保存一帧
        if frame_count % interval == 0:
            # 使用全局帧 ID 作为文件名后缀
            frame_name = f"{prefix}_{global_frame_id:05d}_{modality}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            global_frame_id += 1  # 当前模态的全局帧计数器递增

    cap.release()
    return global_frame_id


def process_annotations_singlemodality(json_path, output_dir, prefix, modality, interval=10, frame_height=1080,
                                       frame_width=1920,
                                       global_image_id=1, global_annotation_id=1):
    """
    处理单模态标注文件，将其转换为 COCO 格式。

    Args:
        json_path (str): 单模态标注文件路径（infrared.json 或 visible.json）。
        output_dir (str): 图像保存路径。
        prefix (str): 视频组名。
        modality (str): 模态类型（visible 或 infrared）。
        interval (int): 每隔多少帧采样。
        frame_height (int): 视频帧高度。
        frame_width (int): 视频帧宽度。
        global_image_id (int): 全局图像 ID 计数器（从 1 开始）。
        global_annotation_id (int): 全局标注 ID 计数器（从 1 开始）。

    Returns:
        tuple: 包括以下两个字段
            - image_info (list): COCO 格式的 images 字段。
            - annotations (list): COCO 格式的 annotations 字段。
            - updated_global_image_id (int): 更新后的全局图像 ID 计数器。
            - updated_global_annotation_id (int): 更新后的全局标注 ID 计数器。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_info = []
    annotations = []

    # 遍历标注文件内容
    for idx, (exist, rect) in enumerate(zip(data['exist'], data['gt_rect'])):
        if idx % interval != 0:  # 跳过非采样帧
            continue

        # 增加全局唯一的 image_id
        image_id = global_image_id
        global_image_id += 1

        # 图像信息（无论目标是否存在都需要记录）
        image_name = f"{prefix}_{image_id:05d}_{modality}.jpg"
        image_info.append({
            "id": image_id,
            "file_name": image_name,
            "height": frame_height,
            "width": frame_width
        })

        # 如果目标不存在
        if exist == 0:
            annotation_id = global_annotation_id
            global_annotation_id += 1
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # 假设 0 表示无目标（无效类别）
                "bbox": [],
                "area": 0,
                "iscrowd": 0
            })
            continue

        # 如果目标存在
        annotation_id = global_annotation_id
        global_annotation_id += 1
        bbox = [rect[0], rect[1], rect[2], rect[3]]  # x, y, width, height
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # 假设只有一个类别 UAV
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })

    # 返回结果和更新后的计数器
    return image_info, annotations, global_image_id, global_annotation_id


def process_annotations_multimodal(json_path_visible, json_path_infrared, output_dir_visible, output_dir_infrared,
                                   prefix, interval=10, height_visible=1080, width_visible=1920, height_infrared=512,
                                   width_infrared=640, global_image_id=1, global_annotation_id=1):
    """
    处理多模态标注文件，将可见光和红外的标注整合为多模态 COCO 格式。

    Args:
        json_path_visible (str): 可见光模态标注文件路径。
        json_path_infrared (str): 红外模态标注文件路径。
        output_dir_visible (str): 可见光模态图像保存路径。
        output_dir_infrared (str): 红外模态图像保存路径。
        prefix (str): 视频组前缀。
        interval (int): 每隔多少帧采样。
        height_visible (int): 可见光图像的高度。
        width_visible (int): 可见光图像的宽度。
        height_infrared (int): 红外图像的高度。
        width_infrared (int): 红外图像的宽度。
        global_image_id (int): 全局图像 ID 计数器（从 1 开始）。
        global_annotation_id (int): 全局标注 ID 计数器（从 1 开始）。

    Returns:
        tuple: 包括以下四个字段
            - image_info (list): COCO 格式的 images 字段（包含多模态路径）。
            - annotations (list): COCO 格式的 annotations 字段。
            - updated_global_image_id (int): 更新后的全局图像 ID 计数器。
            - updated_global_annotation_id (int): 更新后的全局标注 ID 计数器。
    """
    # 读取两个模态的标注文件
    with open(json_path_visible, 'r') as f_visible, open(json_path_infrared, 'r') as f_infrared:
        data_visible = json.load(f_visible)
        data_infrared = json.load(f_infrared)

    image_info = []
    annotations = []

    # 遍历帧标注
    for idx, (exist_v, rect_v, exist_i, rect_i) in enumerate(zip(
            data_visible['exist'], data_visible['gt_rect'],
            data_infrared['exist'], data_infrared['gt_rect']
    )):
        if idx % interval != 0:  # 跳过非采样帧
            continue

        # 创建全局唯一的 image_id
        image_id = global_image_id
        global_image_id += 1

        # 图像信息（无论目标是否存在都需要记录）
        visible_image_name = f"{prefix}_{image_id:05d}_visible.jpg"
        infrared_image_name = f"{prefix}_{image_id:05d}_infrared.jpg"
        image_info.append({
            "id": image_id,
            "file_name_visible": visible_image_name,
            "file_name_infrared": infrared_image_name,
            "height_visible": height_visible,
            "width_visible": width_visible,
            "height_infrared": height_infrared,
            "width_infrared": width_infrared
        })

        # 无目标帧：添加占位符标注
        if exist_v == 0 and exist_i == 0:
            annotation_id = global_annotation_id
            global_annotation_id += 1
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # 假设 0 表示无目标（无效类别）
                "bbox_visible": [],
                "bbox_infrared": [],
                "area_visible": 0,
                "area_infrared": 0,
                "iscrowd": 0
            })
            continue

        # 目标存在帧：添加标注信息
        annotation_id = global_annotation_id
        global_annotation_id += 1
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # 假设 UAV 类别 ID 为 1
            "bbox_visible": rect_v if exist_v == 1 else [],
            "bbox_infrared": rect_i if exist_i == 1 else [],
            "area_visible": rect_v[2] * rect_v[3] if exist_v == 1 else 0,
            "area_infrared": rect_i[2] * rect_i[3] if exist_i == 1 else 0,
            "iscrowd": 0
        })

    # 返回结果和更新后的计数器
    return image_info, annotations, global_image_id, global_annotation_id


def process_dataset(data_dir, output_dir, annotation_output_path, interval=10, is_multimodal=False):
    """
    处理整个数据集：解码视频、降采样帧、整合标注（支持单模态与多模态）。

    Args:
        data_dir (str): UAV_Data 数据文件夹路径（train, val, test）。
        output_dir (str): 保存处理后图像的路径。
        annotation_output_path (str): 输出的 COCO 格式标注文件路径。
        interval (int): 每隔多少帧采样。
        is_multimodal (bool): 是否生成多模态数据集。
    """
    # 全局计数器
    global_image_id = 1
    global_annotation_id = 1

    # 初始化两个模态的全局帧计数器
    global_visible_frame_id = 1
    global_infrared_frame_id = 1

    # 初始化 COCO 数据结构
    images = []
    annotations = []
    categories = [{"id": 1, "name": "UAV", "supercategory": "object"}]  # UAV 类别定义

    # 遍历数据集的二级目录（视频组）
    for video_group in tqdm(os.listdir(data_dir), desc=f"Processing {data_dir}", ncols=100):
        video_group_path = os.path.join(data_dir, video_group)
        if not os.path.isdir(video_group_path):
            continue

        prefix = video_group  # 视频组名作为帧前缀

        if is_multimodal:
            # 多模态：处理 visible 和 infrared
            # 处理可见光视频
            global_visible_frame_id = decode_and_sample(
                video_path=os.path.join(video_group_path, "visible.mp4"),
                output_dir=os.path.join(output_dir, "visible"),
                prefix=prefix,
                modality="visible",
                interval=10,
                global_frame_id=global_visible_frame_id
            )

            # 处理红外视频
            global_infrared_frame_id = decode_and_sample(
                video_path=os.path.join(video_group_path, "infrared.mp4"),
                output_dir=os.path.join(output_dir, "infrared"),
                prefix=prefix,
                modality="infrared",
                interval=10,
                global_frame_id=global_infrared_frame_id
            )

            # 调用 process_annotations_multimodal 处理标注
            img_info, ann, global_image_id, global_annotation_id = process_annotations_multimodal(
                json_path_visible=os.path.join(video_group_path, "visible.json"),
                json_path_infrared=os.path.join(video_group_path, "infrared.json"),
                output_dir_visible=os.path.join(output_dir, "visible"),
                output_dir_infrared=os.path.join(output_dir, "infrared"),
                prefix=prefix,
                interval=interval,
                global_image_id=global_image_id,
                global_annotation_id=global_annotation_id
            )
            images.extend(img_info)
            annotations.extend(ann)
        else:
            # 单模态：默认处理可见光
            # 处理可见光视频
            global_visible_frame_id = decode_and_sample(
                video_path=os.path.join(video_group_path, "visible.mp4"),
                output_dir=os.path.join(output_dir, "visible"),
                prefix=prefix,
                modality="visible",
                interval=10,
                global_frame_id=global_visible_frame_id
            )

            # 调用 process_annotations_singlemodality 处理标注
            img_info, ann, global_image_id, global_annotation_id = process_annotations_singlemodality(
                json_path=os.path.join(video_group_path, "visible.json"),
                output_dir=os.path.join(output_dir, "visible"),
                prefix=prefix,
                modality="visible",
                interval=interval,
                global_image_id=global_image_id,
                global_annotation_id=global_annotation_id
            )
            images.extend(img_info)
            annotations.extend(ann)

    # 保存 COCO 格式的标注文件
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    os.makedirs(os.path.dirname(annotation_output_path), exist_ok=True)
    with open(annotation_output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    base_dir = "./data"  # 原始数据集路径

    # 单模态数据集（可见光）
    process_dataset(
        data_dir=os.path.join(base_dir, "train"),
        output_dir="UAV_COCO_Dataset/train",
        annotation_output_path="UAV_COCO_Dataset/annotations/instances_train_visible.json",
        interval=10,
        is_multimodal=False
    )

    # 多模态数据集
    process_dataset(
        data_dir=os.path.join(base_dir, "train"),
        output_dir="UAV_COCO_Dataset_Multimodal/train",
        annotation_output_path="UAV_COCO_Dataset_Multimodal/annotations/instances_train_multimodal.json",
        interval=10,
        is_multimodal=True
    )
