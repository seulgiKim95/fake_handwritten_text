from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

def open_font_style(ttf_file, size):
    return ImageFont.truetype(ttf_file, size)

def get_single_font_image(char, ttf_file, size):
    font = open_font_style(ttf_file, int(size/3.*2))
    image = Image.new('L', (size, size), color=255) # 'L' mode = single channel
    drawing = ImageDraw.Draw(image)
    w, h = drawing.textsize(char, font=font)
    drawing.text( ((size-w)/2, (size-h)/2), char, fill=(0), font=font)

    flag = np.sum(np.array(image))
    if flag == 255* size*size: # 해당 font에 글자가 없는 경우
        return None
    return image

def merge_image(char, src_img, target_img, size):
    merge_img = Image.new('RGB', (size*2, size), (255,255,255)).convert('L')
    merge_img.paste(target_img, (0, 0)) # 이미지 붙여넣기
    merge_img.paste(src_img, (size, 0))
    return merge_img

def merge_img_array(tgt_arr, src_arr):
    return np.concatenate((tgt_arr, src_arr), axis=1)

def arr_to_img(img_arr):
    img_arr_255 = ((img_arr + 1.) * 127.5)
    img = Image.fromarray(img_arr_255)
    return img
