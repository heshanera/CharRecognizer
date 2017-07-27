# CharRecognizer
Character Recognition using an artificial neural network. Network is trained for 26 characters. Implemented in C++ and Magick++ API is used for pixel manipulation.

### Neural Network
Network is build with 4 layers, Input layer, Output layer and 2 hidden layers with 80 and 40 nodes. Used Sigmoid and Tangent-Sigmoid functions for the node Activation.

### Training 
Network is trained for 26 English uppercase characters from A-Z using 6 training cases for each distinct character.

### Character Recognition
| **Input**  | ![architecture](https://github.com/heshanera/CharRecognizer/blob/master/imgs/testing/C2.png) | ![architecture](https://github.com/heshanera/CharRecognizer/blob/master/imgs/testing/W.png)  | ![architecture](https://github.com/heshanera/CharRecognizer/blob/master/imgs/testing/s.png)  | ![architecture](https://github.com/heshanera/CharRecognizer/blob/master/imgs/testing/A.png)  | ![architecture](https://github.com/heshanera/CharRecognizer/blob/master/imgs/testing/D.png)  |
| ---------- |-----| -----| -----| -----| -----|
| **Output** | `Recognized: C` | `Recognized: W` | `Recognized: S` | `Recognized: A` | `Recognized: D` |
