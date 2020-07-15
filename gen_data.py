import os
import random
import string
from os import path

from captcha import ImageCaptcha

DATASET_PATH = 'dataset'
IMAGE_PATH = 'dataset/images'
LABEL_PATH = 'dataset/labels'

g = ImageCaptcha(
    fonts=[
        'fonts/FreeMono.ttf',
        'fonts/FreeSerif.ttf',
        'fonts/FreeSans.ttf',
    ],
    font_sizes=[42, 52, 62],
    height=80, width=160
)

def gen_sample():
    text = ''.join(random.choices(string.digits, k=4))
    return (text, *g.generate_image(text))

def save_sample(text, image, boxes, name):
    image.save(path.join(IMAGE_PATH, name + '.png'))
    lines = [' '.join([c] + list(map('{:.2f}'.format, b.tolist())))
        for c, b in zip(text, boxes)]
    with open(path.join(LABEL_PATH, name + '.txt'), 'w') as fw:
        fw.write('\n'.join(lines))

def _write_txt(file_path, name_list, base_path, name_f_str):
    with open(file_path, 'w') as fw:
        for i in name_list:
            fw.write(path.join(base_path, name_f_str.format(i)))
            fw.write('\n')

def produce_dataset(N=1000):
    format_str = '{:0' + str(len(str(N))) + '}'
    for i in range(N):
        name = format_str.format(i)
        save_sample(*gen_sample(), name)

    train_len = round(N * 0.8)
    name_list = list(range(N))
    random.shuffle(name_list)
    _write_txt(
        path.join(DATASET_PATH, 'train.txt'),
        name_list[:train_len],
        IMAGE_PATH,
        format_str+'.png'
    )
    _write_txt(
        path.join(DATASET_PATH, 'val.txt'),
        name_list[train_len:],
        IMAGE_PATH,
        format_str+'.png'
    )
    print('gen {} images ({} for train, {} for val).'.format(N, train_len, N-train_len))

if __name__ == "__main__":
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(LABEL_PATH, exist_ok=True)
    produce_dataset(5000)
