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

char Recognizer::recognize(std::string path) {
    
    int w = 40, h = 40; // width x height of a char (in pixels)
    float inputMatrixData[(w*h+1)];
    int *preCharData, *charData;
    ImageProcessor imgPrc;
    
    imgPrc.initializeImage(path);
    imgPrc.createCropedMatrix();
    preCharData = imgPrc.resizeImage();
    charData = imgPrc.skeletonize();
    int brk = 0;
    for (int j = 0; j < (1601); j++) {

        if ( j == 0 ) inputMatrixData[j] = 1; // bias
        else inputMatrixData[j] = charData[j-1]; 
        
        // printing the character ( before skeletonization)
        if (j%40 == 0) std::cout<<"\n";
        if (j < 1600 ) std::cout<<preCharData[j]<<" ";
        
        /*
        // printing the skeletonized Image
        if ( j != 0 ) brk++;
        if (j != 0 ) std::cout<<inputMatrixData[j]<<" ";
        if (brk%40 == 0) std::cout<<"\n";
        */ 
        
        
    }
    inputMatrix.allocateSize(1,1601 /* = width x height + bias = 40*40+1 */);
    inputMatrix.fillMatrix(inputMatrixData);
    
    // loading the data from the data file
    loadWeights();
    // return the output matrix
    getOutputMatrix();
    // return the character
    char character = checkTheRange();
    
    return character;
}

int Recognizer::loadWeights(){
    
    int inputNodes, hiddenLayer1Nodes, hiddenLayer2Nodes, outputNodes;
    int metaData = 0;
    float *weightMatrix1Data, *weightMatrix2Data, *weightMatrix3Data;
    std::string line;
    std::ifstream datafile ("weights");
    
    if (datafile.is_open()) {
        while ( getline (datafile,line) ) {
            
            std::istringstream in(line);
            std::string networkData;
            in >> networkData;
            
            if (networkData == "inputNodes:") {

                in >> inputNodes; metaData++;

            } else if (networkData == "hiddenLayer1Nodes:"){    

                in >> hiddenLayer1Nodes; metaData++;

            } else if (networkData == "hiddenLayer2Nodes:"){    

                in >> hiddenLayer2Nodes; metaData++;

            } else if (networkData == "outputNodes:"){    

                in >> outputNodes; metaData++;

            } else if (networkData == "trainSet:"){    

                in >> trainSet; metaData++;

            }
            if (metaData == 5) break;
        }    
    }    
    
    weightMatrix1Data = new float[ inputNodes * hiddenLayer1Nodes];
    weightMatrix2Data = new float[ hiddenLayer1Nodes * hiddenLayer2Nodes];
    weightMatrix3Data = new float[ hiddenLayer2Nodes * outputNodes];
    
    if (datafile.is_open()) {
        
        while ( getline (datafile,line) ) {
            
            //std::cout << line << '\n';
            std::istringstream in(line);
            std::string networkData;
            in >> networkData;                  
            int tmpindx;
           
            if(networkData == "matrix1:") {
                
                tmpindx = 0;
                int weightMatrix1Size = (inputNodes * hiddenLayer1Nodes); // size of weight matrix 1
                while (tmpindx < weightMatrix1Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix1Data[tmpindx] = weight;
                        tmpindx++;
                }	

            } else if(networkData == "matrix2:") {

                tmpindx = 0;
                int weightMatrix2Size = (hiddenLayer1Nodes * hiddenLayer2Nodes); // size of weight matrix 2    
                while (tmpindx < weightMatrix2Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix2Data[tmpindx] = weight;
                        tmpindx++;
                }	

            } else if(networkData == "matrix3:"){
                
                tmpindx = 0;
                int weightMatrix3Size = (hiddenLayer2Nodes * outputNodes); // size of weight matrix 3  
                while (tmpindx < weightMatrix3Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix3Data[tmpindx] = weight;
                        tmpindx++;
                }
            } else if(networkData == "range:"){
                
                tmpindx = 0;
                
                rangeData = new float[trainSet*2];
                rangeChars = new char[trainSet];
                
                while (tmpindx < ( trainSet ) ) {
                        float rangeS, rangeE;
                        char c;
                        in >> c; in >> rangeS; in >> rangeE;
                        
                        rangeChars[tmpindx] = c;
                        rangeData[(tmpindx*2)] = rangeS;
                        rangeData[(tmpindx*2)+1] = rangeE;
                          
                        //std::cout<<c<<"\t"<<rangeS<<"\t"<<rangeE<<"\n";
                        
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
    //outputLayerMatrix.printMatrix();
    
    
    int rows = outputLayerMatrix.getrows();
    int cols = outputLayerMatrix.getcols();
    /*
    std::cout<<"\n\n-------------------------\n";
    
    std::cout<<"\n\nOut Vector: \n";
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            std::cout<<outputLayerMatrix.get(i,j)*10e4<<" ";
        }
    }
    */
    
    float tmp;
    //std::cout<<"\n\nMedian: \n";
    for (int i = 0; i < rows; i++){
        tmp = 0;
        for (int j = 0; j < cols; j++){
            tmp += outputLayerMatrix.get(i,j);
        }
        tmp = (tmp/cols)*10e4;
        charValue = tmp;
        //std::cout<<(tmp/cols)*10e4<<"\n";
    }
    //std::cout<<"\n\n";
    
    //std::cout<<"\n-------------------------\n\n";
    
    return 0;
}

char Recognizer::checkTheRange(){
    
    std::cout<<"\n\n\n-----------------------------\n\n";
    std::cout<<charValue<<"\t==>";
    
    for (int i = 0; i < trainSet; i++){
        if ( charValue <= rangeData[(i*2)+1] ) {
            
            std::cout<<"\t"<<rangeChars[i];
            std::cout<<"\n\n-----------------------------\n\n\n";
            
            return rangeChars[i];
            break;
        }
    }
    return '\0';
}

int Recognizer::train(int noOfIteration){

    Trainer trainer;
    trainer.initializeWeightMatrices(noOfIteration);
    
    // training for i no of iterations
    for (int i = 0; i < noOfIteration; i++) {    
        trainer.forwardPropagation();
        trainer.backPropagation();
        trainer.printOutputLayer();
    }
    trainer.printdifferenceMeanList();
    trainer.writeWeights();
}