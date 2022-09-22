import numpy as np
from . import images

class Character():
    def __init__(self, font_id, char_id, font_name, char_name, source, target):
        self.font_id = font_id
        self.char_id = char_id
        self.font_name = font_name
        self.char_name = char_name
        self.source = source
        self.target = target

    def draw(self):
        import matplotlib.pyplot as plt
        img = images.merge_img_array(target, source)
        plt.figure()
        plt.imshow(img)
        plt.show()
