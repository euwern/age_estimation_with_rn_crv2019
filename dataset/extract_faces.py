import sys
sys.path.insert(0, 'mtcnn-pytorch')
from src import detect_faces, show_bboxes
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import argparse
import numpy as np

def extract_faces(source_dir, save_dir):

    save_dir = save_dir[:-1] + '_'

    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    for img_file in tqdm(os.listdir(source_dir)):
        
        try:
            image = Image.open(source_dir + img_file)
            image = image.convert('RGB')
            im_array = np.asarray(image)
            if np.mean(im_array) == 0:
                print(source_dir + img_file)
                raise Exception('corrupted image')

        except Exception as e:
            print(e)
            continue

        try:
            bounding_boxes, landmarks = detect_faces(image)
            largest_box = 0
            best_box_ix = 0
            for ix in range(len(bounding_boxes)):
                curr_box = bounding_boxes[ix]
                curr_size = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
                if curr_size > largest_box:
                    largest_box = curr_size
                    best_box_ix = ix

            best_box = bounding_boxes[best_box_ix]
            cropped_img = image.crop((best_box[0], best_box[1], best_box[2], best_box[3]))
            cropped_img.save(save_dir + img_file)
            
        except Exception as e:
            print(e)
            print(img_file)
            #image.save(save_dir + img_file)

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
args = parser.parse_args()
extract_faces(source_dir=args.source_dir, save_dir=args.save_dir)

#dataset_path = '../../dataset/2015'
#extract_faces(source_dir=dataset_path + '/Train/', save_dir=dataset_path + '/TrainFaces/')
#extract_faces(source_dir=dataset_path + '/Validation/', save_dir=dataset_path+'/ValidationFaces/')
#extract_faces(source_dir=dataset_path + '/Test/', save_dir=dataset_path + '/TestFaces/')

#dataset_path = '../../dataset/2016'
#extract_faces(source_dir=dataset_path + '/Train/', save_dir=dataset_path + '/TrainFaces/')
#extract_faces(source_dir=dataset_path + '/Validation/', save_dir=dataset_path+'/ValidationFaces/')
#extract_faces(source_dir=dataset_path + '/Test/', save_dir=dataset_path + '/TestFaces/')

