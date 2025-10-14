import pickle

def file_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))

def file_load(file_path):
    return pickle.load(open(file_path, 'rb'))