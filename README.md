# CharRecognizer
Character Recognition using an artificial neural network. Network is trained for 26 characters. Implemented in C++ and Magick++ API is used for pixel manipulation.

### Neural Network
Network is build with 4 layers, Input layer, Output layer and 2 hidden layers with 80 and 40 nodes. Used Sigmoid and Tangent-Sigmoid functions for the node Activation.

### Training 
Network is trained for 26 English uppercase characters from A-Z using 6 training cases for each distinct character.
<hr>

### Input
![architecture](https://github.com/heshanera/CharRecognizer/blob/master/imgs/testing/C2.png)

### Output

0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 0 <br>
0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 <br>
0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 <br>
0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 <br>
0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 <br>
0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 <br>
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 <br>
1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 <br>
0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 <br>
0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 <br>
0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 0 <br>
0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 1 0 0 <br>
0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 <br>

- ##### Recognized: C

