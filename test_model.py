from keras.models import load_model
import numpy as np
import sys
import pickle

#load text
text_ints = np.load('saved_data/int_book_text.npy')

#load dict for converting ints back to chars
with open('saved_data/int_to_vocab_dict.pkl', 'rb') as f:
    int_to_char_dict = pickle.load(f)

#slightly modified function from book_generator_lstm.py
def generate_text(model, num_of_characters):

    start_prime_index = np.random.randint(0, len(text_ints)-1)
    text = text_ints[start_prime_index].tolist()

    print("Generating text from model file: {}".format(sys.argv[1]))
    print("Starting seed is: '{}' ".format(''.join([int_to_char_dict[char] for char in text])))
    print()

    for i in range(num_of_characters):
        x = np.reshape(text, (1, len(text), 1))
        x = x / float(len(int_to_char_dict))
        prediction = model.predict(x, verbose=0)

        index = np.argmax(prediction)
        result = int_to_char_dict[index]
        seq_in = [int_to_char_dict[value] for value in text]
        sys.stdout.write(result)
        sys.stdout.flush()
        text.append(index)
        text = text[1:len(text)]

if sys.argv[1] == "" or sys.argv[2] == "":
    print("Usage: python test_model.py /path/to/model number_of_characters_to_generate")
    sys.exit(1)

model = load_model(sys.argv[1])
generate_text(model, sys.argv[2])
