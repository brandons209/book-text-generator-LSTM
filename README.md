# Generate new text for a book!
[LICENSE](LICENSE)

### Objective
The purpose of this project is to generate new unique text based on trained text from a book. Although I used a book, it can be trained with any type of text. Feel free to use multiple articles from a write, songs, or even tweets!

### Libraries
I used Keras with the Tensorflow backend to create and train the model and TensorBoard for visualization.

### Model Architecture
This model is an LSTM Recurrent Neural Network, with two LSTM layers for 600 nodes each, with a classifier layer to predict the next character. It was tried using the Adam optimizer and categorical crossentropy as the loss.

### Performance
With a training loss around 1.1 and 70 percent accuracy, the model is able to generate semi-coherent sentences. It can use quotes properly, punctuation, and most of the words make some sense. It does misspell quite often. However, without embedding layers and just generating text based on trained characters, it would need very long training times to get better. I trained this model on 150,000 characters from the 1 million or so characters in the Harry Potter book included, with a sequence length of 100. Even with that and limiting the vocabulary size, it still took 8 hours and 60 epochs to get the performance I am on a NVIDIA 1060 GPU. If I were to train the entire text over the course of 3-4 days, I could get much better results.

### Running the model
1. Set the sequence length in book_generator_lstm.py, the lower the better text generation, but it increases training time.
    * If you change the sequence length, you may want to change the number of epochs and batch size as well.
2. Set the book path for the book you want to use. I have it set to harry_potter_book_4.txt. I also include a book called Anna under anna.txt.
3. Set the amount of text to take out from the book to train on, I used the first 150,000 characters for the Harry Potter book. You can also comment this out to use the entire text.
4. (Optional) If you want to see TensorBoard graphs, run tensorboard --logdir=tensorboard_logs
5. Run book_generator_lstm.py. It will process the text file with the sequence length and set everything up. After training, it will generate text based on a random seed from the book with the best weights loaded.
6. Finally, you will be asked whether to delete the weight files under saved_weights.

### Testing the model
Once you have ran your model, it will be saved under the start time of the program in saved_models. You can run test_model.py to generate your choice amount of characters with the chosen saved model. It will pick a random seed from the book loaded under saved_data. Even if you are not training and using the model I provide for the Harry Potter book, you need to run book_generator_lstm.py up until before training the model so it can generate the text in integers and dictionaries for testing the model.

Usage:
```shell
python test_model.py /path/to/model number_of_characters_to_generate
```
### Final Notes
With more training, the results can be improved. I just don't have the time to leave my computer training for days. I will also probably make a version of this project using embedded vectors for the words as well. Good luck and I hope you can learn from my code!
