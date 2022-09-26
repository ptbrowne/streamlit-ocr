import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import sys

def process(t):
    lines = '\n'.join([re.sub('[0-9],?', '', l) for l in t.split('\n') if l != ''])
    return lines

def main():
    filename = '/tmp/t.jpg'
    img = np.array(Image.open(filename))

    lang = sys.argv[1]
    text = pytesseract.image_to_string(img, lang=lang)
    print(process(text))

if __name__ == '__main__':
    main()