/* 
 * File:   Trainer.cpp
 * Author: heshan
 * 
 * Created on May 6, 2017, 5:37 PM
 */

#include <iostream>

#include "Trainer.h"
#include "Matrix.h"
#include "Activation.h"
#include "ImageProcessor.h"

Trainer::Trainer() { }

Trainer::Trainer(const Trainer& orig) { }

Trainer::~Trainer() { }

int Trainer::initializeWeightMatrices() { 
    
    classes = 26; // output node classes
    chars = 8; // number of training chars 
    int w = 40, h = 40; // width x height of a char (in pixels)
    int size = 1600; // width x height
    learningRate = 0.5; // learning rate of the network 
    
    // Initializing the input Matrix **************************************************************/
    std::string trainingImages[] = {"imgs/training/A.jpg","imgs/training/B.jpg","imgs/training/C.jpg",
                                    "imgs/training/D.jpg","imgs/training/E.jpg","imgs/training/F.jpg",
                                    "imgs/training/G.jpg","imgs/training/H.jpg"};
    ImageProcessor imgPrc;
    float inputMatrixData[(w*h+1)*chars];
    int* tmpData;
    
    for (int i = 0; i < chars; i++) {
        
        imgPrc.initializeImage(trainingImages[i]);
        imgPrc.createCropedMatrix();
        tmpData = imgPrc.resizeImage();
        
        for (int j = 0; j < (1601); j++) {
            
            if ( j == 0 ) inputMatrixData[(1601*i)] = 1; // bias
            else inputMatrixData[j + (1601*i)] = tmpData[j-1]; 
            
            
            //std::cout<<inputMatrixData[j+ (1600*i)]<<" ";
            //if (j%40 == 0) std::cout<<"\n";
        }
        //std::cout<<"\n\n";
    }    
    inputMatrix.allocateSize(chars,1601 /* = width x height + bias = 40*40+1 */);
    inputMatrix.fillMatrix(inputMatrixData);
    //inputMatrix.printMatrix();
    
    // Initializing the weight Matrix1 **************************************************************/
    hiddenLayer1Nodes = 500;
    int charPixSize = 1601 /* = width x height + bias = 40*40+1 */;
    float randomFloat;
    float weightMatrix1Data[ charPixSize * hiddenLayer1Nodes];
    int k = 0;
    for(int i = 0; i < charPixSize; i++){
        for(int j = 0; j < hiddenLayer1Nodes; j++){
            randomFloat = (static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 50));
            weightMatrix1Data[k] = randomFloat; k++;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix1.allocateSize(charPixSize, hiddenLayer1Nodes);
    weightMatrix1.fillMatrix(weightMatrix1Data);
    
    // Initializing the weight Matrix2 **************************************************************/
    hiddenLayer2Nodes = 750;
    float weightMatrix2Data[ hiddenLayer1Nodes * hiddenLayer2Nodes];
    k = 0;
    for(int i = 0; i < hiddenLayer1Nodes; i++){
        for(int j = 0; j < hiddenLayer2Nodes; j++){
            randomFloat = (static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 50));
            weightMatrix2Data[k] = randomFloat; k++;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix2.allocateSize(hiddenLayer1Nodes, hiddenLayer2Nodes);
    weightMatrix2.fillMatrix(weightMatrix2Data);
    
    // Initializing the weight Matrix3 **************************************************************/
    int outputLayerNodes = 26;
    float weightMatrix3Data[ hiddenLayer2Nodes * outputLayerNodes ];
    k = 0;
    for(int i = 0; i < hiddenLayer2Nodes; i++){
        for(int j = 0; j < outputLayerNodes; j++){
            randomFloat = (static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 50));
            weightMatrix3Data[k] = randomFloat; k++;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix3.allocateSize(hiddenLayer2Nodes,outputLayerNodes);
    weightMatrix3.fillMatrix(weightMatrix3Data);
    //weightMatrix3.printMatrix();
    
    // Initializing the target Matrix **************************************************************/
    float targetMatrixData[classes*chars];
    k = 0;
    for (int i = 0; i < chars; i++){
        for (int j = 0; j < classes; j++){
            if ( i == j ) targetMatrixData[k] = 1;
            else targetMatrixData[k] = 0;
            k++;
        }
   
    }
    targetMatrix.allocateSize(chars,classes);
    targetMatrix.fillMatrix(targetMatrixData);
    
    return 0;
}

int Trainer::forwardPropagation(){
    
    // input layer --> hidden layer 1
    hiddenLayer1Matrix = inputMatrix.matrixMul(weightMatrix1);
    hiddenLayer1Matrix = Activation::sigmoid(hiddenLayer1Matrix);
    //hiddenLayer1Matrix.printMatrix();
    
    
    // hidden layer 1 --> hidden layer 2    
    hiddenLayer2Matrix = hiddenLayer1Matrix.matrixMul(weightMatrix2);
    hiddenLayer2Matrix = Activation::sigmoid(hiddenLayer2Matrix);
    //hiddenLayer2Matrix.printMatrix();
    
    
    // hidden layer 2 --> output layer
    outputLayerMatrix = hiddenLayer2Matrix.matrixMul(weightMatrix3);
    outputLayerMatrix = Activation::sigmoid(outputLayerMatrix);
    //outputLayerMatrix.printMatrix();
    
    return 0;
}

int Trainer::backPropagation(){

    
    // updating weight matrix 3 ( hidden layer 2 --> output layer )
    w3Delta1 = outputLayerMatrix.subtract(targetMatrix);
    w3Delta2 = outputLayerMatrix.hadamardMul(outputLayerMatrix.subtractFrom(1));
    w3Delta3 = hiddenLayer2Matrix.transpose();        
    w3Delta = w3Delta3.matrixMul(w3Delta1.hadamardMul(w3Delta2)).scalarMul(learningRate);      
    
    // updating weight matrix 2 ( hidden layer 1 --> hidden layer 2 )
    w2Delta1 = w3Delta1.hadamardMul(w3Delta2).matrixMul(weightMatrix3.transpose());
    w2Delta2 = hiddenLayer2Matrix.hadamardMul(hiddenLayer2Matrix.subtractFrom(1));
    w2Delta3 = hiddenLayer1Matrix.transpose().matrixMul(w2Delta2);
    w2Delta.allocateSize(hiddenLayer1Nodes,hiddenLayer2Nodes);
    
    for (int i = 0; i < hiddenLayer1Nodes; i++){
        for (int j = 0; j < hiddenLayer2Nodes; j++){
            w2Delta.set(i, j, ( w2Delta3.get(i,j)*w2Delta1.get(0,j)*learningRate ));
        }
    }
    
    // updating weight matrix 1 ( input layer 1 --> hidden layer 1 )
    
    
    
    
    // updating the weights
    weightMatrix3.subtract(w3Delta);
    weightMatrix2.subtract(w2Delta);
    
    
    return 0;
}
