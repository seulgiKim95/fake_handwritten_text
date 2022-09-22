from get_data import font2img
from common import utils
from PIL import ImageFont
from PIL import Image
import glob
import numpy as np
from scipy.misc import imresize
import pickle
from get_data import char_class


src_path = 'C:/Users/user/Desktop/GAN-handwriting-styler-master/GAN_project/data/font_cgh/source/'
trg_path = 'C:/Users/user/Desktop/GAN-handwriting-styler-master/GAN_project/data/font_cgh/target/'
output_path = 'C:/Users/user/Desktop/GAN-handwriting-styler-master/GAN_project/data/result_cgh/'



# 여기는 font2img 부분
# src_file = glob.glob(src_path+'*')[0]
# trg_files = glob.glob(trg_path+'*')
# hangul = []
# with open('get_data/2350-common-hangul.txt','r',encoding='utf-8') as f:
#     hangul = [line.rstrip() for line in f]

# src_font = ImageFont.truetype(src_file,64)
# dst_font = ImageFont.truetype(trg_files[0],64)
# for font in trg_files:
#     font_name = font.split('\\')[-1].split('.')[0]
#     for char in hangul:
#         x = font2img.draw_example(char, src_font, dst_font, 128)
#         x.save('./data/result_cgh/'+font_name+'_'+char+'.jpg')





# 요기부터 전처리
def normalize_image(img_arr): # 글씨 부분은 -1, 배경은 1로 normalize
    normalized = img_arr / 127.5 -1.
    return normalized


def cropping(img_arr):
    img_size = img_arr.shape[0]
    full_white = np.asarray(Image.new('L', (img_size, img_size), color=255)).astype(np.float)
    col_sum = np.where(np.sum(full_white, axis=0) - np.sum(img_arr, axis=0) > 1)
    row_sum = np.where(np.sum(full_white, axis=1) - np.sum(img_arr, axis=1) > 1)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = img_arr[y1:y2, x1:x2]
    return cropped_image

def resizing(img_arr, max_size, resize_fix=False):
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

def merge_img_array(tgt_arr, src_arr):
    return np.concatenate((tgt_arr, src_arr), axis=1)

def arr_to_img(img_arr):
    img_arr_255 = ((img_arr + 1.) * 127.5)
    img = Image.fromarray(img_arr_255)
    return img


######### main ##########
size = 128
font_size = 120

hangul = []
with open('get_data/2350-common-hangul.txt','r',encoding='utf-8') as f:
    hangul = [line.rstrip() for line in f]
outputs = glob.glob(output_path+'*')
font_list = list(set([n.split('\\')[-1].split('_')[0] for n in outputs]))

data_list = []

for i in outputs:
    font_name = i.split('\\')[-1].split('_')[0]
    char_name = i.split('\\')[-1].split('_')[-1].split('.')[0]

    src_img, tgt_img = utils.read_split_image(i)

    src_arr = np.asarray(src_img).astype(np.float)
    tgt_arr = np.asarray(tgt_img).astype(np.float)

    crop_src = cropping(src_arr)
    resize_src = resizing(crop_src, font_size, resize_fix=10.)
    pad_src = padding(resize_src, size)

    crop_tgt = cropping(tgt_arr)
    resize_tgt = resizing(crop_tgt, font_size, resize_fix=10.)
    pad_tgt = padding(resize_tgt, size)

    merged = merge_img_array(pad_tgt, pad_src)
    pil_image = arr_to_img(merged)
    pil_image = pil_image.convert('L')
    pil_image.save('./data/final_data/final_'+font_name+'_'+char_name+'.jpg')

    test = char_class.Char(font_list.index(font_name), hangul.index(char_name), font_name, char_name, pad_src, pad_tgt)
    data_list.append(test)

with open('./data/final_pickle/test.pickle', 'wb') as f:
    pickle.dump(data_list, f)