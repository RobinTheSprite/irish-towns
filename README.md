# irish-towns
A neural network for generating fictional Irish town names

This was a project for CS605, Artificial Intelligence, taught by Dr. Jon Genetti at the University of Alaska Fairbanks.

The goal was to create a neural network that was capable of generating fake Irish town names. That is, town names that have a characteristically Irish spelling, but do not actually exist. At first, a GAN was constructed with the intent of performing self-training. This network failed to produce good enough output, so a far simpler LSTM network was built instead.

## Data
The training data for the project was pulled directly from [this page on Wikipedia](https://en.wikipedia.org/wiki/List_of_towns_and_villages_in_the_Republic_of_Ireland). The data was simplified by lowercasing all letters, replacing the '-' character with spaces, and removing apostrophes. The remaining set of characters in the list were the English alphabet a-z, the five Irish "fada" vowels (á, é, í, ó, ú), the space character ' ', and a special "start" character '#'. 

The data was then broken up into n-grams of customizable length. Before gramifying the name, a number of start characters equal to the length of the n-gram was prepended to the name. Thus, the beginning of every name started with n '#'s. This allowed the LSTM to use the start characters as a seed value when generating names.

Spaces were also appended to the end of approximately one-third of the names, in order to train the network to stop generating letter characters after a while. This allowed the network to generate names of realistic length.

## Network

The network itself was two LSTM layers with Dropout layers in between, and one final Dense layer. The network was trained on the n-grams prepared previously, and try to predict what the n+1-th character would be. A one-hot vector containing the probability of each character being the next one is the output. The most likely character is chosen, according to a certain temperature value.

## Results

The LSTM network was reasonably capable of generating realistic Irish town names. Notable examples include ballymacarslew, dounery, ballymeal, ballybridge, kilmeeran, and donnelly. Note that Donnelly is an Irish last name, but *was not in the training data.*
