import pandas as pd
import numpy as np
import os
import shutil
import math

def move_file(file, src, dest):
    src_path = os.path.join(src, file)
    dest_path = os.path.join(dest, file)
    shutil.move(src_path, dest_path)


def create_folder(path, name):
    folder_name = os.path.join(path, name)
    os.mkdir(folder_name)


# split data into train and test - return list of file names
def split_files(Arr, name, percentage=0.8):
    rows = Arr.shape[0]
    indices = np.arange(rows)
    permuted = np.random.permutation(indices)
    p = math.ceil(rows*percentage)
    return Arr[permuted[:p],], Arr[permuted[p:],]


def fill_folder(folder_name, files):
    for file in files:
        move_file(file+'.jpg', data_ISIC, folder_name)


# set paths
work_path = os.path.abspath(os.getcwd())
data_ISIC = os.path.join(work_path, 'ISIC_2019_Training_Input')
ground_truth_file = os.path.join(work_path, 'ISIC_2019_Training_GroundTruth.csv')


# create data folder and inside train and test folders
create_folder(work_path, 'data')
data_path = os.path.join(work_path, 'data')

create_folder(data_path, 'train')
data_train_path = os.path.join(data_path, 'train')

create_folder(data_path, 'test')
data_test_path = os.path.join(data_path, 'test')


# load information about the dataset
df = pd.read_csv(ground_truth_file)

# remove unneccesary column
df = df.iloc[:, :-1]

# function to get data easily
get_data = lambda df, col: df[df[column] == 1].iloc[:, 0].to_numpy()

# iterate over all cancer types - avoid column filename (image)
for_train = for_test = 0
for column in df.columns[1:]:
    train, test = split_files(get_data(df, column), column)
    
    for_train, for_test = for_train + len(train), for_test + len(test)
    print(f'Train: {len(train)} - Test {len(test)}')
    
    create_folder(data_train_path, column)
    fill_folder(os.path.join(data_train_path, column), train)
    
    create_folder(data_test_path, column)
    fill_folder(os.path.join(data_test_path, column), test)

print(f'Nº Train data: {for_train}\nNº Test data: {for_test}')

# remove folder ISIC
shutil.rmtree(data_ISIC)
