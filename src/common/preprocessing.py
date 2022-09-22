import numpy as np
from PIL import Image
from scipy.misc import imresize # scipy.__version__ == 1.2.0

def normalize_image(img_arr):
    # image의 value 0~255를 -1~1로 normalization
    # 글씨 부분은 -1, 배경은 1
    normalized = img_arr / 127.5 -1.
    return normalized

def cropping(img_arr):
    # 배경이 아닌 글자 부분을 사각형으로 자르기
    img_size = img_arr.shape[0]
    full_white = np.asarray(Image.new('L', (img_size, img_size), color=255)).astype(np.float)
    col_sum = np.where(np.sum(full_white, axis=0) - np.sum(img_arr, axis=0) > 1)
    row_sum = np.where(np.sum(full_white, axis=1) - np.sum(img_arr, axis=1) > 1)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = img_arr[y1:y2, x1:x2]
    return cropped_image

def resizing(img_arr, max_size, resize_fix=False):
    # 글자 사이즈 조절
    # max_size : 폰트 조절될 크기
    if type(resize_fix) == int: # 큰 쪽을 resize_fix로 고정
        origin_h, origin_w = img_arr.shape
        if origin_h > origin_w:
            resize_w = int(origin_w * (resize_fix / origin_h))
            resize_h = resize_fix
        else:
            resize_h = int(origin_h * (resize_fix / origin_w))
            resize_w = resize_fix
    elif type(resize_fix) == float: # resize_fix 만큼 곱하기
        origin_h, origin_w = img_arr.shape
        resize_h, resize_w = int(origin_h * resize_fix), int(origin_w * resize_fix)
        if resize_h > max_size: # max_size보다 글씨가 크면 max_size로 고정
            resize_h = max_size
            resize_w = int(resize_w * max_size / resize_h)
        if resize_w > max_size:
            resize_w = max_size
            resize_h = int(resize_h * max_size / resize_w)

    # resize
    if resize_fix != False:
        img_arr = imresize(img_arr, (resize_h, resize_w))
        img_arr = normalize_image(img_arr)

    return img_arr

def padding(img_arr, size, pad_value=None):
    # 전체 사이즈를 size로 조정
    # pad value : 패딩에 넣을 값 지정
    height, width = img_arr.shape
    if not pad_value:
        pad_value = img_arr[0][0]

    # Adding padding of x axis
    pad_x_width = (size - width) // 2
    pad_x = np.full( (height, pad_x_width), pad_value, dtype=np.float32)
    img_arr = np.concatenate( (pad_x, img_arr), axis=1)
    img_arr = np.concatenate( (img_arr, pad_x), axis=1)
    width = img_arr.shape[1]

    # Adding padding of y axis
    pad_y_height = (size - height) // 2
    pad_y = np.full( (pad_y_height, width), pad_value, dtype=np.float32)
    img_arr = np.concatenate((pad_y, img_arr), axis=0)
    img_arr = np.concatenate((img_arr, pad_y), axis=0)

    # match to original image size
    width = img_arr.shape[1]
    if img_arr.shape[0] % 2: # where height is size-1
        pad = np.full( (1, width), pad_value, dtype=np.float32)
        img_arr = np.concatenate((pad, img_arr), axis=0)
    height = img_arr.shape[0]
    if img_arr.shape[1] % 2: # where width is size-1
        pad = np.full( (height, 1), pad_value, dtype=np.float32)
        img_arr = np.concatenate( (pad, img_arr), axis=1)

    return img_arr
