"""
该脚本用于从实验获得的图片中读取数据
图片名必须类似 001_2023-05-16_14-38-00.png 的格式
"""
import re
import argparse
import pandas as pd
from cnocr import CnOcr
from PIL import Image
import os

from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从实验获得的图片中读取数据')
    parser.add_argument('-in', '--in_path', type=str, default='this_dir', help='输入路径')
    parser.add_argument('-out', '--out_path', type=str, default='desktop', help='输出路径')
    _args = parser.parse_args()

    if _args.in_path == 'this_dir':
        _args.in_path = os.getcwd()
        print('[info] 未指定输入路径，使用当前目录。')
        print(f'[info] path: {_args.in_path}')

    if _args.out_path == 'desktop':
        _args.out_path = os.path.join(os.path.expanduser('~'), "Desktop")
        print('[info] 未指定输出路径，使用桌面。')
        print(f'[info] path: {_args.out_path}')

    return _args


def parse_img(_img_path: str, rec_name='en_PP-OCRv3'):
    """解析单张图片中的数据"""
    img = Image.open(_img_path)  # 打开图片
    # 切图 crop(left, upper, right, lower)
    img_crop = {
        'mau_set_freq': img.crop((860, 1380, 950, 1410)),  # MAU 设定频率
        'ahu_set_freq': img.crop((860, 1460, 950, 1485)),  # AHU 设定频率
        'ef_set_freq': img.crop((860, 1530, 950, 1560)),  # EF 设定频率
        'mau_freq': img.crop((766, 1162, 887, 1191)),  # MAU 频率
        'ahu_freq': img.crop((1509, 1162, 1626, 1191)),  # AHU 频率
        'ef_freq': img.crop((2431, 1162, 2553, 1191)),  # EF 频率
        'fresh_volume': img.crop((1124, 679, 1217, 709)),  # 新风量
        'supply_volume': img.crop((1610, 865, 1710, 894)),  # 送风量
        'return_volume': img.crop((1612, 682, 1712, 711)),  # 回风量
        'exhaust_volume': img.crop((2374, 782, 2468, 809)),  # 排风量
        'room1_pres': img.crop((1382, 1450, 1480, 1480)),  # 一更室内压差
        'room2_pres': img.crop((1554, 1450, 1654, 1480)),  # 二更室内压差
        'room3_pres': img.crop((2242, 1450, 2412, 1480)),  # 洁净走廊室内压差
        'room4_pres': img.crop((1726, 1450, 1823, 1480)),  # 测试间一室内压差
        'room5_pres': img.crop((1895, 1450, 1995, 1480)),  # 测试间二室内压差
        'room6_pres': img.crop((2067, 1450, 2167, 1480)),  # 测试间三室内压差
        'room1_hepa_pres': img.crop((1390, 1529, 1480, 1556)),  # 一更 HEPA 压差
        'room2_hepa_pres': img.crop((1554, 1529, 1654, 1556)),  # 二更 HEPA 压差
        'room3_hepa_pres_a': img.crop((2243, 1529, 2338, 1556)),  # 洁净走廊 1 HEPA 压差
        'room3_hepa_pres_b': img.crop((2400, 1529, 2503, 1556)),  # 洁净走廊 2 HEPA 压差
        'room4_hepa_pres': img.crop((1726, 1529, 1823, 1556)),  # 测试间一 HEPA 压差
        'room5_hepa_pres': img.crop((1895, 1529, 1995, 1556)),  # 测试间二 HEPA 压差
        'room6_hepa_pres': img.crop((2067, 1529, 2167, 1556)),  # 测试间三 HEPA 压差
    }

    final_data = {}
    ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name=rec_name)

    # 识别图片中的文字
    for k, v in img_crop.items():
        out = ocr.ocr(v)
        final_data.update({k: out[0]['text']})

    return final_data


def parse(path_in: str, path_out: str):
    imgs_dir = path_in  # 图片目录
    datas = []  # 读取的数据

    # 获取 imgs_dir 目录下文件的文件名
    file_names = os.listdir(imgs_dir)
    # 遍历图片文件
    for file_name in tqdm(file_names):
        # 校验文件名，必须类似 001_2023-05-16_14-38-00.png 的格式
        if not re.match(r'^\d{3}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.png$', file_name):
            continue
        img_path = os.path.join(imgs_dir, file_name)  # 图片路径
        data = parse_img(img_path)  # 解析图片中的数据
        data.update({'uid': file_name[:3]})  # 将图片 uid 添加到数据中
        datas.append(data)  # 将数据添加到 datas 列表中

    df = pd.DataFrame(datas)
    # 排序并保存数据
    columns = ['uid', 'mau_set_freq', 'ahu_set_freq', 'ef_set_freq', 'mau_freq', 'ahu_freq',
               'ef_freq', 'fresh_volume', 'supply_volume', 'return_volume', 'exhaust_volume',
               'room1_pres', 'room2_pres', 'room3_pres', 'room4_pres', 'room5_pres', 'room6_pres',
               'room1_hepa_pres', 'room2_hepa_pres', 'room3_hepa_pres_a', 'room3_hepa_pres_b',
               'room4_hepa_pres', 'room5_hepa_pres', 'room6_hepa_pres']
    df[columns].to_excel(os.path.join(path_out, 'ParsedData.xlsx'), index=False)

    pass


if __name__ == '__main__':
    args = parse_args()
    parse(args.in_path, args.out_path)
