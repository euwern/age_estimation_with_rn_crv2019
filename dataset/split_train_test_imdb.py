from random import shuffle

file = open('../dataset/imdb_crop/data.csv', 'r')

dataset = file.readlines()

n_dataset = len(dataset)
index = list(range(n_dataset))
shuffle(index)

train_file = open('../dataset/imdb_crop/Train.csv', 'w')
test_file = open('../dataset/imdb_crop/Test.csv', 'w')

for ix in range(n_dataset):
    rand_ix = index[ix]
    if ix < n_dataset * 0.1:
        test_file.write(dataset[rand_ix])
    else:
        train_file.write(dataset[rand_ix])

file.close()
train_file.close()
test_file.close()
