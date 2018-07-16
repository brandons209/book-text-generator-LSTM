#general imports:
import numpy as np
import sys, os
import time
from glob import glob

#model imports
from keras.layers import Dropout, Dense
from keras.layers import LSTM
from keras.models import Sequential

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

#data processing and other helper functions
import helper_functions as helper

#name of the book text file to be used for training:
book_name = 'harry_potter_book_4.txt'

#length of each sequence for input data
seq_length = 100

#get book text converted to ints, their targets, with a dict to convert the ints back to text, and a dict to convert characters to ints
train_text, train_targets, int_to_char_dict, char_to_int_dict = helper.read_book(book_name, seq_length)

#number of classes is going to be the length of the vocab, in this case the int_dict, used for one hot encoding
num_classes = len(int_to_char_dict)
print("Number of vocabulary: {}".format(num_classes))

#normalize input data to floats from 0 to 1
#can also one-hot encode these instead, might get better results.
#save raw ints to generate text later
train_text_raw = train_text
train_text = train_text.astype('float32')/num_classes

# reshape data to be in format of: [samples, time steps, features] THIS HAS TO BE DONE FOR ALL LSTMS in keras
# data is already in format [samples, sequence length]
# so this would be [number of characters, length of sequence, 1] 1 for features since each character is 1 feature that is inputed into the network
train_text = np.reshape(train_text, (len(train_text), seq_length, 1))

#trained on the first 150,000 characters since training on the ~1 million characters in the book would take a long time.
train_text, train_targets = train_text[0:150000], train_targets[0:150000]
#one hot encode targets: i put it here instead of one hot encoding first because numpy.delete was flattening the array.
train_targets = helper.one_hot_encode(train_targets, num_classes)

#dont need testing and valid set, since we are looking for the model to generalize the text, not predict exactly the next character.
print("Training character size is: {:,}.".format(len(train_text)))
print("Data shape is: {}".format(train_text.shape))
print("Sequence length is {} and number of features for lstm input is 1.".format(seq_length))

#Model creation: use lstm cells with a softmax fully connected output layer.
model = Sequential()
#lstm layers:
#input shape is (seq_length, 1)
model.add(LSTM(600, input_shape=(train_text.shape[1], train_text.shape[2]), return_sequences=True))#return_sequences to true since we have a second LSTM layer
model.add(Dropout(0.5))
model.add(LSTM(600))
model.add(Dropout(0.5))
#prediction layer:
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#compile model with adam optimizer and loss as categorical_crossentropy, since we are using one hot encoded catergories.
model.compile(optimizer=opt.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#hyperparameters
epochs = 60
batch_size = 64

input("Press enter to train the model...")
#train Model

#checkpoint callback to save best weights, each one saved with epoch number and train loss
#we dont use validation set because we are trying to minmize training loss to best generalize the text to create its own sensible new text, not copy the train text.
#just have to avoid overfitting.
checkpointer = ModelCheckpoint(filepath='saved_weights/weights.epoch-{epoch:02d}.loss-{loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=True)
#tensorboard with logs saved to tensorboard_logs/start_time To see tensorboard info, run tensorboard --logdir=tensorboard_logs
start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())
ten_board = TensorBoard(log_dir="tensorboard_logs/{}".format(start_time), write_images=True)

model.fit(train_text, train_targets, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer, ten_board])

#load best weights from saved weights, which would be the last one saved.
weight_files = os.listdir('saved_weights/')
print("Loading best weights from saved_weights...")
model.load_weights(weight_files[len(weight_files)-1])

#save model
save_dir = 'saved_models/book_gen_model_{}.h5'.format(start_time)
print("Saving model to {}.".format(save_dir))
model.save(save_dir)
input("Press enter to generate text from model...")

#generates number of characters from model as from the argument. it picks a random seed from the train text to start.
def generate_text(num_of_characters):
    start_prime_index = np.random.randint(0, len(train_text)-1)
    text = train_text_raw[start_prime_index].tolist()

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

generate_text(2000)

#clear weights if user desires.
answer = input("Would you like to delete all of the weight files? (y or n)\n")
if answer == 'y':
    files = os.listdir('saved_weights/')
    for file in files:
        if file.endswith(".hdf5"):
            print("Deleting {}".format(file))
            os.remove(os.path.join('saved_weights/', file))
