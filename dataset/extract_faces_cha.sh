#assuming you have downloaded chalearn 2015 and 2016 dataset and unzip and renamed the files accordingly.

python extract_faces_wiki.py --source_dir '../../dataset/2015/Train/' --save_dir '../../dataset/2015/TrainFaces/'&
python extract_faces_wiki.py --source_dir '../../dataset/2015/Validation/' --save_dir '../../dataset/2015/ValidationFaces/'&
python extract_faces_wiki.py --source_dir '../../dataset/2015/Test/' --save_dir '../../dataset/2015/TestFaces/'&

python extract_faces_wiki.py --source_dir '../../dataset/2016/Train/' --save_dir '../../dataset/2016/TrainFaces/'&
python extract_faces_wiki.py --source_dir '../../dataset/2016/Validation/' --save_dir '../../dataset/2016/ValidationFaces/'&
python extract_faces_wiki.py --source_dir '../../dataset/2016/Test/' --save_dir '../../dataset/2016/TestFaces/'&

