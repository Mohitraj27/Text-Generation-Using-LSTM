# Text-Generation-Using-LSTM

## About the Project
The poem by Robert Frost "Stopping by words on the Snowing Evening" was converted from PDF to text, pre-processed, and tokenized using word indexes. It was transformed into 50-word sequences. An LSTM model with a vocabulary size and vector space of 50 was trained for 200 epochs, achieving 84.83 accuracy. 
![lstm](https://github.com/Mohitraj27/Text-Generation-Using-LSTM/assets/87956374/addf0791-2551-4953-8d63-eead3e8ab6d9)


### Automatic Text Generation
Automatic text generation is the generation of natural language texts by computer. It has applications in automatic documentation systems, automatic letter writing, automatic report generation, etc. In this project, we are going to generate words given a set of input words. We are going to train the LSTM model using Robert Frost poem Stopping by woods in a snowing evenining which is taken in pdf format and then converted into text file.

## Dataset Used
### Download Link
```
https://drive.google.com/file/d/1ZdSH2PJZROCHjz3PYKd7AtnzicbxWSh1/view?usp=sharing
```
```
Stopping by Woods on a Snowy Evening Robert Frost  
 
 Whose woods these are I think I know  
His house is in the village though  
He will not see me stopping here  
To watch his woods fill up with snow  
 
My little horse must think it queer  
To stop without a farmhouse near  
Between the woods and frozen lake  
The darkest evening of the year  
 
He gives his harness bells a shake  
To ask if there is some mistake  
The only other sounds the sweep  
Of easy wind and downy flake  
 
The woods are lovely dark and deep  
But I have promises to keep  
And miles to go before I sleep  
And miles to go before I sleep  
  Admiring Light on a Sunny Day Erika Fitzpatrick  
 
What light this is I may it know  
Its beams barred by finite time though  
He should not mind me pausing now  
To admire this light ere it go  
 
My wearied mind considers how  
There is time enough to allow  
Dead and dilated eyes to gaze  
On light thats not for me endowed  
 
It filters through in timid haze  
For this room its not seen in days  
Dust dances where lighted  day glows  
In mute music and golden rays  
 
Sunlight is happy hope arose  
But I have ssignments to close  
And pages to rove before I doze  
And pages to rove before I doze  
```
### Long Short Term Memory Network (LSTM)
- Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory.
- Generally LSTM is composed of a cell (the memory part of the LSTM unit) and three "regulators", usually called gates, of the flow of information inside the LSTM unit: an input gate, an output gate and a forget gate.
- Intuitively, the cell is responsible for keeping track of the dependencies between the elements in the input sequence.
- The input gate controls the extent to which a new value flows into the cell, the forget gate controls the extent to which a value remains in the cell and the output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit.
- The activation function of the LSTM gates is often the logistic sigmoid function.
- There are connections into and out of the LSTM gates, a few of which are recurrent. The weights of these connections, which need to be learned during training, determine how the gates operate.

### Architecture of LSTM
![lstm_working](https://github.com/Mohitraj27/Text-Generation-Using-LSTM/assets/87956374/39a10d23-32df-4162-985a-9403b3a5c0b6)

## Pre- Processing of Dataset
![image](https://github.com/Mohitraj27/Text-Generation-Using-LSTM/assets/87956374/b66e6156-d0e8-41d6-b5fc-7637dd284e2a)

### Removing Stopwards
![image](https://github.com/Mohitraj27/Text-Generation-Using-LSTM/assets/87956374/4504f123-1902-457a-b7b1-d6593257a54e)

### Removing Lemmatization
![image](https://github.com/Mohitraj27/Text-Generation-Using-LSTM/assets/87956374/c3ebb2e2-a469-4f26-9d41-52f423fb6211)

#### Converting texts in form of Tokens
```
['stopping', 'by', 'woods', 'on', 'a', 'snowy', 'evening', 'robert', 'frost', 'whose', 'woods', 'these', 'are', 'i', 'think', 'i', 'know', 'his', 'house', 'is', 'in', 'the', 'village', 'though', 'he', 'will', 'not', 'see', 'me', 'stopping', 'here', 'to', 'watch', 'his', 'woods', 'fill', 'up', 'with', 'snow', 'my', 'little', 'horse', 'must', 'think', 'it', 'queer', 'to', 'stop', 'without', 'a']
```
#### Converting Tokens into text into sequences
Create a unique numerical token for each unique word in the dataset. fit_on_texts() updates internal vocabulary based on a list of texts. texts_to_sequences() transforms each text in texts to a sequence of integers.sequences containes a list of integer values created by tokenizer. Each line in sequences has 51 words. Now we will split each line such
that the first 50 words are in X and the last word is in y.

```
array([112,  36,  11,  17,   9, 136,  35, 134, 133, 132,  11, 131,  33,
         4,  34,   4,  31,  18, 126,   7,  13,   3, 122,  29,  15, 118,
         8, 116,  14, 112, 113,   1, 110,  18,  11, 106, 104, 103, 101,
        27, 100,  98,  97,  34,   6,  94,   1,  92,  90,   9])
```
 Vocab Size is 137 for my dataset
 Sequence Length is of 50


 ### LSTM Model
 A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

 #### Embedding Layer

 The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset. It requires 3 arguments:

- input_dim: This is the size of the vocabulary in the text data which is vocab_size in this case.
- output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word.
- input_length: Length of input sequences which is seq_length.

#### LSTM Layer
This is the main layer of the model. It learns long-term dependencies between time steps in time series and sequence data. return_sequence when set to True returns the full sequence as the output.

#### Dense Layer


