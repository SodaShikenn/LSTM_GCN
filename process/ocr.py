from paddleocr import PaddleOCR
import pandas as pd
from glob import glob
import os
import cv2
from tqdm import tqdm
import logging

# 屏蔽调试错误
logging.disable(logging.DEBUG)

class OCR():
    def __init__(self):
        self.ocr = PaddleOCR()

    def scan(self, file_path, output_path, marked_path=None):
        # 文字识别
        info = self.ocr.predict(file_path)
        print(info)

if __name__ == '__main__':
    ocr = OCR()
    ocr.scan("../input/imgs/train/34908612.jpeg", None)