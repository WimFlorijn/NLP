from Project.dataset import build_data_set

if __name__ == '__main__':
    print('Downloading data set from Twitter...')
    data_set = build_data_set()
    print('Downloading complete.')
