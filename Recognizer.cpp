/* 
 * File:   Recognizer.cpp
 * Author: heshan
 * 
 * Created on May 4, 2017, 8:50 PM
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <Magick++.h>
#include "ImageProcessor.h"
#include "Recognizer.h"
#include "Activation.h"
#include "Matrix.h"
#include "Trainer.h"

Recognizer::Recognizer() { }

Recognizer::Recognizer(const Recognizer& orig) { }

Recognizer::~Recognizer() { }

int Recognizer::recognize(std::string path) {
    
    int w = 40, h = 40; // width x height of a char (in pixels)
    float inputMatrixData[(w*h+1)];
    int* charData;
    ImageProcessor imgPrc;
    
    imgPrc.initializeImage(path);
    imgPrc.createCropedMatrix();
    charData = imgPrc.resizeImage();
    int brk = 0;
    for (int j = 0; j < (1601); j++) {

        if ( j == 0 ) inputMatrixData[j] = 1; // bias
        else inputMatrixData[j] = charData[j-1]; 

        if ( j != 0 ) brk++;
        if (j != 0 ) std::cout<<inputMatrixData[j]<<" ";
        if (brk%40 == 0) std::cout<<"\n";
    }
    inputMatrix.allocateSize(1,1601 /* = width x height + bias = 40*40+1 */);
    inputMatrix.fillMatrix(inputMatrixData);
    
    // loading the data from the data file
    loadWeights();
    // return the output matrix
    getOutputMatrix();
    
    return 0;
}

int Recognizer::loadWeights(){
    
    int inputNodes = 1601;
    int hiddenLayer1Nodes = 500;
    int hiddenLayer2Nodes = 750;
    int outputNodes = 26;
    
    float weightMatrix1Data[ inputNodes * hiddenLayer1Nodes];
    float weightMatrix2Data[ hiddenLayer1Nodes * hiddenLayer2Nodes];
    float weightMatrix3Data[ hiddenLayer2Nodes * outputNodes];
    
    std::string line;
    std::ifstream datafile ("weights");
    
    if (datafile.is_open()) {
        
        while ( getline (datafile,line) ) {
            //std::cout << line << '\n';
            std::istringstream in(line);
            std::string matrix;
            in >> matrix;                  

            int tmpindx = 0;
            if(matrix == "matrix1:") {
                
                int weightMatrix1Size = (inputNodes * hiddenLayer1Nodes); // size of weight matrix 1
                while (tmpindx < weightMatrix1Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix1Data[tmpindx] = weight;
                        tmpindx++;
                }	

            } else if(matrix == "matrix2:") {

                int weightMatrix2Size = (hiddenLayer1Nodes * hiddenLayer2Nodes); // size of weight matrix 2    
                while (tmpindx < weightMatrix2Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix2Data[tmpindx%4] = weight;
                        tmpindx++;
                }	

            } else if(matrix == "matrix3:"){
                
                int weightMatrix3Size = (hiddenLayer2Nodes * outputNodes); // size of weight matrix 3  
                while (tmpindx < weightMatrix3Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix3Data[tmpindx%4] = weight;
                        tmpindx++;
                }
            }
        }
        datafile.close();
    }

    else std::cout << "Unable to load the data file";

    weightMatrix1.allocateSize(inputNodes,hiddenLayer1Nodes);	
    weightMatrix2.allocateSize(hiddenLayer1Nodes,hiddenLayer2Nodes);
    weightMatrix3.allocateSize(hiddenLayer2Nodes,outputNodes);
    
    weightMatrix1.fillMatrix(weightMatrix1Data);
    weightMatrix2.fillMatrix(weightMatrix2Data);
    weightMatrix3.fillMatrix(weightMatrix3Data);
    
}


int Recognizer::getOutputMatrix(){
    
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
    outputLayerMatrix.printMatrix();
    
    return 0;
}

int Recognizer::train(){

    Trainer trainer;
    trainer.initializeWeightMatrices();
    
    // training for i no of iterations
    for (int i = 0; i < 20; i++) {    
        trainer.forwardPropagation();
        trainer.backPropagation();
        //trainer.printOutputLayer();
    }
    trainer.writeWeights();
}