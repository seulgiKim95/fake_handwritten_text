#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import common
import pickle
warnings.filterwarnings('ignore')


SRC_DIR = '../data/source/'
TRG_DIR = '../data/target/'
target_list = ['Batang-01.ttf', 'Dotum-03.ttf', 'Gulim-01.ttf', 'Gungsuh-03.ttf', 'H2GPRM.TTF', 'H2GTRE.TTF', 'H2HDRM.TTF']
OUTPUT_DIR = '../result/'
IMAGE_DIR = OUTPUT_DIR + 'font_images/'
pickle_file = OUTPUT_DIR + 'fonts.pickle'

# initialize
size = 128 # canvas size
font_size = 110
# size = 64
# font_size = 50

source = glob.glob(SRC_DIR+'*')[0]
targets = [TRG_DIR+tgt for tgt in target_list] # glob.glob(TRG_DIR+'*')[:7]

char_file = '../data/2350-common-hangul.txt'
with open(char_file, 'r') as rf:
    hangul2350 = [line.rstrip() for line in rf]

font_names = [ target.split('/')[-1].split('.')[0] for target in targets ]

chars = []
for i in range(len(targets)):
    target = targets[i]
    font_name = font_names[i]
    for ch in hangul2350:
        src_img = common.images.get_single_font_image(ch, source, size)
        tgt_img = common.images.get_single_font_image(ch, target, size)
        merged_img = common.images.merge_image(ch, src_img, tgt_img, size)
 
        src_arr = np.asarray(src_img).astype(np.float)
        tgt_arr = np.asarray(tgt_img).astype(np.float)
 
        crop_src = common.preprocessing.cropping(src_arr)
        resize_src = common.preprocessing.resizing(crop_src, font_size, resize_fix=10.)
        pad_src = common.preprocessing.padding(resize_src, size)

        crop_tgt = common.preprocessing.cropping(tgt_arr)
        resize_tgt = common.preprocessing.resizing(crop_tgt, font_size, resize_fix=10.)
        pad_tgt = common.preprocessing.padding(resize_tgt, size)
 
        chars.append(common.classes.Character(font_names.index(font_name), hangul2350.index(ch), font_name, ch, pad_src, pad_tgt))


# save pickle
with open(pickle_file, 'wb') as wf:
    pickle.dump(chars, wf)


# save figure
for font in font_names: # making directory to save images
    os.makedirs(IMAGE_DIR+font, exist_ok=True)
for char in chars: # save jpg
    merged = common.images.merge_img_array(char.target, char.source)
    img = common.images.arr_to_img(merged).convert('L')
    img.save(IMAGE_DIR + char.font_name + '/' + char.char_name+'.png')
