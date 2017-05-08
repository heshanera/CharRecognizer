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
    chars = 25; // number of training chars 
    int w = 40, h = 40; // width x height of a char (in pixels)
    int size = 1600; // width x height
    learningRate = 0.01; // learning rate of the network 
    inputLayerNodes = size + 1;
    
    // Initializing the input Matrix **************************************************************/
    std::string trainingImages[] = {"imgs/training/A.jpg","imgs/training/B.jpg","imgs/training/C.jpg",
                                    "imgs/training/D.jpg","imgs/training/E.jpg","imgs/training/F.jpg",
                                    "imgs/training/G.jpg","imgs/training/H.jpg",//"imgs/training/I.jpg",
                                    "imgs/training/J.jpg","imgs/training/K.jpg","imgs/training/L.jpg",
                                    "imgs/training/M.jpg","imgs/training/N.jpg","imgs/training/O.jpg",
                                    "imgs/training/P.jpg","imgs/training/Q.jpg","imgs/training/R.jpg",
                                    "imgs/training/S.jpg","imgs/training/T.jpg","imgs/training/U.jpg",
                                    "imgs/training/V.jpg","imgs/training/W.jpg","imgs/training/X.jpg",
                                    "imgs/training/Y.jpg","imgs/training/Z.jpg"};
    ImageProcessor imgPrc;
    float inputMatrixData[(w*h+1)*chars];
    int* tmpData;
    
    for (int i = 0; i < chars; i++) {
        
        imgPrc.initializeImage(trainingImages[i]);
        imgPrc.createCropedMatrix();
        tmpData = imgPrc.resizeImage();
        int brk = 0;
        for (int j = 0; j < (1601); j++) {
            
            if ( j == 0 ) inputMatrixData[(1601*i)] = 1; // bias
            else inputMatrixData[j + (1601*i)] = tmpData[j-1]; 
            
            if ( j != 0 ) brk++;
            if (j != 0 ) std::cout<<inputMatrixData[j + (1601*i)]<<" ";
            if (brk%40 == 0) std::cout<<"\n";
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
            randomFloat = 0.0085;//LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));//(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 40));
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
            randomFloat = 0.009;//LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));//(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 40));
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
            randomFloat = 0.0087;//LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));//(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 40));
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
    w1Delta1 = w2Delta1.matrixMul(weightMatrix2.transpose());
    w1Delta2 = hiddenLayer1Matrix.hadamardMul(hiddenLayer1Matrix.subtractFrom(1));
    w1Delta3 = inputMatrix.transpose().matrixMul(w1Delta2);
    w1Delta.allocateSize(inputLayerNodes,hiddenLayer1Nodes);
    for (int i = 0; i < inputLayerNodes; i++){
        for (int j = 0; j < hiddenLayer1Nodes; j++){
            w1Delta.set(i, j, ( w1Delta3.get(i,j)*w1Delta1.get(0,j)*learningRate ));
        }
    }
    
    // updating the weights
    weightMatrix3 = weightMatrix3.subtract(w3Delta);
    weightMatrix2 = weightMatrix2.subtract(w2Delta);
    weightMatrix1 = weightMatrix1.subtract(w1Delta);
    //weightMatrix3.printMatrix();
    //weightMatrix2.printMatrix();
    
    
    return 0;
}

int Trainer::printOutputLayer(){
    
    std::cout<<"\n\ntarget Matrix\n";
    targetMatrix.printMatrix();
    std::cout<<"\n\noutput Matrix\n";
    int rows = outputLayerMatrix.getrows();
    int cols = outputLayerMatrix.getcols();
    float tmp = 0;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            tmp = outputLayerMatrix.get(i,j)*100000;
            std::cout<<tmp<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n\nfinal Matrix\n";
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            tmp = outputLayerMatrix.get(i,j)*100000;
            if ( tmp < 4.23047 ) std::cout<<0<<" ";
            else std::cout<<1<<" ";         
        }
        std::cout<<"\n";
    }
    
    return 0;        
}