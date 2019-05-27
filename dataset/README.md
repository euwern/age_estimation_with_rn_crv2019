## Data Preparation:
1. Please clone https://github.com/TropComplique/mtcnn-pytorch because we use mtcnn to detect faces.
2. Download the ChaLearn 2015, 2016, WIKI, and IMDB dataset accordingly.
    * ChaLearn 2015: http://chalearnlap.cvc.uab.es/challenge/12/description/
    * ChaLearn 2016: http://chalearnlap.cvc.uab.es/dataset/19/description/
    * IMDB and WIKI dataset: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
3. Run the split train test script for IMDB and WIKI
4. Extract faces from respective dataset by running shell scripts.
5. Thrash imdb and wiki faces that are too small (filter_faces_imdb.py and filter_faces_wiki.py)
