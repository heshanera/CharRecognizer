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
    
    int w = 20, h = 20; // width x height of a char (in pixels)
    float inputMatrixData[(w*h+1)];
    int *preCharData, *charData;
    ImageProcessor imgPrc;
    int inputArraySize = w*h +1
    imgPrc.initializeImage(path);
    imgPrc.createCropedMatrix();
    /*preCharData*/charData = imgPrc.resizeImage();
    //charData = imgPrc.skeletonize();
    int brk = 0;
    for (int j = 0; j < inputArraySize; j++) {

        if ( j == 0 ) inputMatrixData[j] = 1; // bias
        
        else {
            if (charData[j-1] == 1) inputMatrixData[j]  = 1;
            else inputMatrixData[j]  = -1;
        }
        
        // printing the character ( before skeletonization)
        if (j%40 == 0) std::cout<<"\n";
        if (j < 1600 ) std::cout<<charData[j]<<" ";
        
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
    char character = '\0';//checkTheRange();
    
    return character;
}

int Recognizer::loadWeights(){
    
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

            } else if (networkData == "distinctChars:"){    

                in >> distinctChars; metaData++;

            } else if (networkData == "distinctCharList:"){    

                trainedChars = new std::string[distinctChars];
                std::string tmpChar;
                for(int i = 0; i < distinctChars; i++){
                    in >> tmpChar;
                    trainedChars[i] = tmpChar;
                }
                metaData++;
                
            }
            if (metaData == 7) break;
        }    
    }    
    
    weightMatrix1Data = new float[ inputNodes * hiddenLayer1Nodes];
    weightMatrix2Data = new float[ hiddenLayer1Nodes * hiddenLayer2Nodes];
    weightMatrix3Data = new float[ hiddenLayer2Nodes * outputNodes];
    
    weightMatrix1List = new float*[distinctChars]; for(int i = 0; i < distinctChars; i++) weightMatrix1List[i] = new float[inputNodes*hiddenLayer1Nodes];
    weightMatrix2List = new float*[distinctChars]; for(int i = 0; i < distinctChars; i++) weightMatrix2List[i] = new float[hiddenLayer1Nodes*hiddenLayer2Nodes];
    weightMatrix3List = new float*[distinctChars]; for(int i = 0; i < distinctChars; i++) weightMatrix3List[i] = new float[hiddenLayer2Nodes*outputNodes];
    
    if (datafile.is_open()) {
        
        int readMatrices = 0;
        int charIndex = 0;
        //std::string chars[] = {"A","B","C","D"};
        
        while ( getline (datafile,line) ) {
            
            //std::cout << line << '\n';
            std::istringstream in(line);
            std::string networkData;
            in >> networkData;                  
            int tmpindx;
            
            if(networkData == (trainedChars[charIndex]+"_matrix1:")) {

                tmpindx = 0;
                int weightMatrix1Size = (inputNodes * hiddenLayer1Nodes); // size of weight matrix 1
                while (tmpindx < weightMatrix1Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix1List[charIndex][tmpindx] = weight;
                        tmpindx++;
                }	
                readMatrices++;

            } else if(networkData == (trainedChars[charIndex]+"_matrix2:")) {

                tmpindx = 0;
                int weightMatrix2Size = (hiddenLayer1Nodes * hiddenLayer2Nodes); // size of weight matrix 2    
                while (tmpindx < weightMatrix2Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix2List[charIndex][tmpindx] = weight;
                        tmpindx++;
                }	
                readMatrices++;

            } else if(networkData == (trainedChars[charIndex]+"_matrix3:")) {

                tmpindx = 0;
                int weightMatrix3Size = (hiddenLayer2Nodes * outputNodes); // size of weight matrix 3  
                while (tmpindx < weightMatrix3Size ) {
                        float weight;
                        in >> weight;
                        weightMatrix3List[charIndex][tmpindx] = weight;
                        tmpindx++;
                }
                readMatrices++;
            }
            if ( readMatrices == 3 ) { readMatrices = 0; charIndex++; }
        }        
        datafile.close();
    }

    else std::cout << "Unable to load the data file";
 
}


int Recognizer::getOutputMatrix(){
    
    weightMatrix1.allocateSize(inputNodes,hiddenLayer1Nodes);	
    weightMatrix2.allocateSize(hiddenLayer1Nodes,hiddenLayer2Nodes);
    weightMatrix3.allocateSize(hiddenLayer2Nodes,outputNodes);
    
    int rows, cols;
    
    std::cout<<"\n\n-------------------------\n";
    //std::cout<<"\n\nOut Vector: \n";
    
    std::string tmpChars = "";
    float tmpMax = 0;
    
    for (int i = 0; i < distinctChars; i++){
    
        weightMatrix1.fillMatrix(weightMatrix1List[i]);
        weightMatrix2.fillMatrix(weightMatrix2List[i]);
        weightMatrix3.fillMatrix(weightMatrix3List[i]);
        
        // input layer --> hidden layer 1
        hiddenLayer1Matrix = inputMatrix.matrixMul(weightMatrix1);
        hiddenLayer1Matrix = Activation::tanSigmoid(hiddenLayer1Matrix);
        //hiddenLayer1Matrix.printMatrix();


        // hidden layer 1 --> hidden layer 2    
        hiddenLayer2Matrix = hiddenLayer1Matrix.matrixMul(weightMatrix2);
        hiddenLayer2Matrix = Activation::sigmoid(hiddenLayer2Matrix);
        //hiddenLayer2Matrix.printMatrix();


        // hidden layer 2 --> output layer
        outputLayerMatrix = hiddenLayer2Matrix.matrixMul(weightMatrix3);
        outputLayerMatrix = Activation::sigmoid(outputLayerMatrix);
        //outputLayerMatrix.printMatrix();

        rows = outputLayerMatrix.getrows();
        cols = outputLayerMatrix.getcols();
        
        float tmpW;
        for (int k = 0; k < rows; k++){
            tmpW = 0;
            for (int j = 0; j < cols; j++){
                tmpW += outputLayerMatrix.get(k,j);
            }
            tmpW /= outputNodes;
            //std::cout<<tmpW<<" ";
        } 
        /*
        if (tmpW > 0.4965) tmpChars += trainedChars[i];
        else tmpChars += "*";
        */
        if (tmpW > tmpMax) {
            tmpMax = tmpW;
            tmpChars = trainedChars[i];
        }    
        
    }
    std::cout<<"# Recognized: "<<tmpChars;
    std::cout<<"\n-------------------------\n\n";
    
    return 0;
}

int Recognizer::train(int noOfIteration){

    Trainer trainer;
    /*
    trainer.initializeWeightMatrices(noOfIteration);
    int iterationNo = 0;
    // training for i no of iterations
    for (int i = 0; i < noOfIteration; i++) {
        
        iterationNo++;
        std::cout<<"Iteration: "<<iterationNo<<"\n";
        
        trainer.forwardPropagation();
        trainer.backPropagation();
        trainer.printOutputLayer();
    }
    //trainer.printdifferenceMeanList();
     * 
     * 
     */
    trainer.train(noOfIteration);
    //trainer.writeWeights();
}