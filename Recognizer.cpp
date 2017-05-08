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
#include "Recognizer.h"
#include "ImageProcessor.h"
#include "Trainer.h"
#include "Matrix.h"

Recognizer::Recognizer() { }

Recognizer::Recognizer(const Recognizer& orig) { }

Recognizer::~Recognizer() { }

int Recognizer::recognize(std::string path) {
    
    Matrix inputMatrix;
    float inputMatrixData[(w*h+1)];
    int* charData;
    ImageProcessor imgPrc;
    
    imgPrc.initializeImage(path);
    imgPrc.createCropedMatrix();
    charData = imgPrc.resizeImage();
    int brk = 0;
    for (int j = 0; j < (1601); j++) {

        if ( j == 0 ) inputMatrixData[(1601*i)] = 1; // bias
        else inputMatrixData[j + (1601)] = charData[j-1]; 

        if ( j != 0 ) brk++;
        if (j != 0 ) std::cout<<inputMatrixData[j + (1601)]<<" ";
        if (brk%40 == 0) std::cout<<"\n";
    }
    inputMatrix.allocateSize(1,1601 /* = width x height + bias = 40*40+1 */);
    inputMatrix.fillMatrix(inputMatrixData);
    
}

int Recognizer::loadWeights(){
    
    int inputNodes = 1601;
    int hiddenLayer1Nodes = 500;
    int hiddenLayer2Nodes = 750;
    int outputNodes = 26;
    
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
                        data2[tmpindx] = weight;
                        tmpindx++;
                }	

            } else if(matrix == "matrix2:") {

                int weightMatrix2Size = (hiddenLayer1Nodes * hiddenLayer2Nodes); // size of weight matrix 2    
                while (tmpindx < weightMatrix2Size ) {
                        float weight;
                        in >> weight;
                        data3[tmpindx%4] = weight;
                        tmpindx++;
                }	

            } else if(matrix == "matrix3:"){
                
                int weightMatrix3Size = (hiddenLayer2Nodes * outputNodes); // size of weight matrix 3  
                while (tmpindx < weightMatrix3Size ) {
                        float weight;
                        in >> weight;
                        data3[tmpindx%4] = weight;
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
    
    weightMatrix1.fillMatrix(data2);
    weightMatrix2.fillMatrix(data3);
    weightMatrix3.fillMatrix(data3);
    
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