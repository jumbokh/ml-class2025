# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:48:08 2020

@author: ThinkPad
"""

import numpy as np
from PIL import Image


class SVD:
    
    def __init__(self, img_path):
        with Image.open(img_path) as img:
            img = np.asarray(img.convert('L'))
        self.U, self.Sigma, self.VT = np.linalg.svd(img)
    
    def compress_img(self, k: "# singular value") -> "img":
        return self.U[:, :k] @ np.diag(self.Sigma[:k]) @ self.VT[:k,:]
        

if __name__ == '__main__':
    model = SVD('lenna.jpg')
    result = Image.fromarray(model.compress_img(10))