import os
from tqdm import tqdm
from PIL import Image

source_dir = '../../dataset/wiki/faces/'
thrash_dir = '../../dataset/wiki/faces_thrash/'
if not os.path.exists(thrash_dir):
    os.makedirs(thrash_dir)
total_count = 0 
small_images = 0
for img_file in tqdm(os.listdir(source_dir)):
    im = Image.open(source_dir + img_file)
    width, height = im.size
    if width < 32 or height < 32:
        os.rename(source_dir + img_file, thrash_dir + img_file)
        small_images += 1
    total_count += 1

print(small_images)
print(total_count)
