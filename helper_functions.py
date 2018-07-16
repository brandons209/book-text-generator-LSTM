import numpy as np
from keras.utils import np_utils
from tqdm import tqdm
import pickle
import os

#takes path to book text file: loads text and converts it to use with network, with shape (len(book_text), seq_length). since each entry is seq_length of characters
#returns dictionaries to convert text converted into integers back to text, integers of the text in sequences, and their targets.
def read_book(book_path, seq_length):
    save_file_names = ['saved_data/int_book_text.npy', 'saved_data/targets.npy', 'saved_data/int_to_vocab_dict.pkl',
                       'saved_data/vocab_to_int_dict.pkl']
    file_counter = 0
    for file in save_file_names:
            if os.path.isfile(file):
                file_counter += 1
    if file_counter == 4:
        print("Saved files found in saved_data, loading those files...")
        return np.load(save_file_names[0], mmap_mode='r'), np.load(save_file_names[1], mmap_mode='r'), pickle.load(open(save_file_names[2], 'rb')), pickle.load(open(save_file_names[3], 'rb'))

    with open(book_path, 'r') as book:
        book_text = book.read()
    #convert to lowercase so have fewer vocab, remove -- from text
    book_text = book_text.lower()
    book_text = book_text.replace("--", "")

    vocab_dict = sorted(set(book_text))
    vocab_to_int_dict = {char: int for int, char in enumerate(vocab_dict)}
    int_to_vocab_dict = dict(enumerate(vocab_dict))

    #Each training pattern of the network is comprised of 100 time steps of one character - int_book_text, followed by one character output - targets
    int_book_text = []
    targets = []
    print("Loading text from file and targets...")
    for i in tqdm(range(0, len(book_text) - seq_length, 1)):
        seq_in = book_text[i:i + seq_length]
        seq_out = book_text[i + seq_length]
        int_book_text.append([vocab_to_int_dict[char] for char in seq_in])
        targets.append(vocab_to_int_dict[seq_out])

    int_book_text = np.array(int_book_text, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    print("Saving int_book_text, targets, and dicts to saved_data/")
    np.save(save_file_names[0], int_book_text)
    np.save(save_file_names[1], targets)
    with open(save_file_names[2], 'wb') as f:
        pickle.dump(int_to_vocab_dict, f)
    with open(save_file_names[3], 'wb') as f:
        pickle.dump(vocab_to_int_dict, f)

    return int_book_text, targets, int_to_vocab_dict, vocab_to_int_dict

#one hot encodes data with num_classes.
def one_hot_encode(data, num_classes):
    return np_utils.to_categorical(data, num_classes)
