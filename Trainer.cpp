/* 
 * File:   Trainer.cpp
 * Author: heshan
 * 
 * Created on May 6, 2017, 5:37 PM
 */

#include <iostream>
#include <fstream>
#include "Trainer.h"
#include "Matrix.h"
#include "Activation.h"
#include "ImageProcessor.h"

Trainer::Trainer() { }

Trainer::Trainer(const Trainer& orig) { }

Trainer::~Trainer() { }

int Trainer::initializeWeightMatrices(int noOfIteration) { 
    
    classes = 25; // output node classes
    chars = 27; // number of training chars 
    int w = 40, h = 40; // width x height of a char (in pixels)
    int size = 1600; // width x height
    learningRate = 0.5; // learning rate of the network 
    differenceMeanList = new float[noOfIteration]; //No of training Iterations
    iterationNo = 0;
    
    inputLayerNodes = size + 1;
    hiddenLayer1Nodes = 150;
    hiddenLayer2Nodes = 300;
    
    
    // Initializing the input Matrix **************************************************************/
    std::string trainingImages[] = {"imgs/training/A.jpg","imgs/training/A2.jpg","imgs/training/A3.jpg","imgs/training/B.jpg","imgs/training/C.jpg",
                                    "imgs/training/D.jpg","imgs/training/E.jpg","imgs/training/F.jpg",
                                    "imgs/training/G.jpg","imgs/training/H.jpg",//"imgs/training/.jpg",
                                    "imgs/training/J.jpg","imgs/training/K.jpg","imgs/training/L.jpg",
                                    "imgs/training/M.jpg","imgs/training/N.jpg","imgs/training/O.jpg",
                                    "imgs/training/P.jpg","imgs/training/Q.jpg","imgs/training/R.jpg",
                                    "imgs/training/S.jpg","imgs/training/T.jpg","imgs/training/U.jpg",
                                    "imgs/training/V.jpg","imgs/training/W.jpg","imgs/training/X.jpg",
                                    "imgs/training/Y.jpg","imgs/training/Z.jpg"};
    
    char typeOfTrainingChars[] = {  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 
                                    11 ,12 ,13, 14, 15, 16, 17, 18, 19, 20,
                                    21, 22, 23, 24, 25};
    
    
    char caps[] = { 'A','B','C','D','E','F','G','H',//'I',
                    'J','K','L','M','N','O','P','Q','R',
                    'S','T','U','V','W','X','Y','Z'};
    
    targetChars = new char[chars];
    for (int i = 0; i < chars; i++){
        targetChars[i] = caps[typeOfTrainingChars[i]-1];
    }
    
    ImageProcessor imgPrc;
    float inputMatrixData[(w*h+1)*chars];
    int *tmpData, *tmpData2;
    
    for (int i = 0; i < chars; i++) {
        
        imgPrc.initializeImage(trainingImages[i]);
        imgPrc.createCropedMatrix();
        tmpData2 = imgPrc.resizeImage();
        tmpData = imgPrc.skeletonize();
                
        //if (i == 0 ) tmpData = tmp1;       
        //else tmpData = tmp2;
        
        int brk = 0;
        for (int j = 0; j < inputLayerNodes; j++) {
            
            if ( j == 0 ) inputMatrixData[(inputLayerNodes*i)] = 1; // bias
            else inputMatrixData[j + (inputLayerNodes*i)] = tmpData[j-1]; 
            
            if ( j != 0 ) brk++;
            //if (j != 0 ) std::cout<<inputMatrixData[j + (1601*i)]<<" ";
            if (j != 0 ) std::cout<<tmpData2[j-1]<<" ";
            if (brk%40 == 0) std::cout<<"\n";
        }
        //std::cout<<"\n\n";
    }    
    inputMatrix.allocateSize(chars,inputLayerNodes);
    inputMatrix.fillMatrix(inputMatrixData);
    //inputMatrix.printMatrix();
    
    float LO = 0.001, HI = 0.049;
    
    // Initializing the weight Matrix1 **************************************************************/
    
    int charPixSize = inputLayerNodes;
    float randomFloat;
    float weightMatrix1Data[ charPixSize * hiddenLayer1Nodes];
    int k = 0;
    for(int i = 0; i < charPixSize; i++){
        for(int j = 0; j < hiddenLayer1Nodes; j++){
            randomFloat = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));//(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 40));
            weightMatrix1Data[k] = randomFloat; k++;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix1.allocateSize(charPixSize, hiddenLayer1Nodes);
    weightMatrix1.fillMatrix(weightMatrix1Data);
    
    // Initializing the weight Matrix2 **************************************************************/
    float weightMatrix2Data[ hiddenLayer1Nodes * hiddenLayer2Nodes];
    k = 0;
    for(int i = 0; i < hiddenLayer1Nodes; i++){
        for(int j = 0; j < hiddenLayer2Nodes; j++){
            randomFloat = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));//(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 40));
            weightMatrix2Data[k] = randomFloat; k++;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix2.allocateSize(hiddenLayer1Nodes, hiddenLayer2Nodes);
    weightMatrix2.fillMatrix(weightMatrix2Data);
    
    // Initializing the weight Matrix3 **************************************************************/
    int outputLayerNodes = classes;
    float weightMatrix3Data[ hiddenLayer2Nodes * outputLayerNodes ];
    k = 0;
    for(int i = 0; i < hiddenLayer2Nodes; i++){
        for(int j = 0; j < outputLayerNodes; j++){
            randomFloat = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));//(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 40));
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
            if ( typeOfTrainingChars[i]-1 == j ) targetMatrixData[k] = 1;
            else targetMatrixData[k] = 0;
            k++;
        }
   
    }
    targetMatrix.allocateSize(chars,classes);
    targetMatrix.fillMatrix(targetMatrixData);
    targetMatrix.printMatrix();
    
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

int Trainer::writeWeights(){
    
    std::ofstream weightData;
    weightData.open ("weights");

    weightData <<"inputNodes: "<<inputLayerNodes;
    weightData <<"\nhiddenLayer1Nodes: "<<hiddenLayer1Nodes;
    weightData <<"\nhiddenLayer2Nodes: "<<hiddenLayer2Nodes;
    weightData <<"\noutputNodes: "<<classes;
    weightData <<"\ntrainSet: "<<chars;
    
    weightData <<"\n\n";
    
    weightData <<"\nmatrix1: ";
    int rows = weightMatrix1.getrows();
    int cols = weightMatrix1.getcols();
    for (int i = 0; i < rows; i++) {	
        for (int j = 0; j < cols; j++) {	
            float weight = weightMatrix1.get(i,j);
            weightData <<weight<<" ";
        }    
    }	
    weightData <<"\nmatrix2: ";
    rows = weightMatrix2.getrows();
    cols = weightMatrix2.getcols();
    for (int i = 0; i < rows; i++)
    {	
    for (int j = 0; j < cols; j++)
        {	
            float weight = weightMatrix2.get(i,j);
            weightData <<weight<<" ";
        }    
        //weightData <<"\n";
    }	
    weightData <<"\nmatrix3: ";
    rows = weightMatrix3.getrows();
    cols = weightMatrix3.getcols();
    for (int i = 0; i < rows; i++)
    {	
    for (int j = 0; j < cols; j++)
        {	
            float weight = weightMatrix3.get(i,j);
            weightData <<weight<<" ";
        }    
        //weightData <<"\n";
    }	
    weightData <<"\n\n\n";
    
    // writing the character ranges
    weightData <<"range: ";
    for(int i = 0; i < chars; i++){
        weightData <<rangeChars[i]<<" ";
        weightData <<rangeData[i*2]<<" ";
        weightData <<rangeData[(i*2)+1]<<" ";
    }
    
    weightData.close();
    
    return 0;
}

int Trainer::printOutputLayer(){
    
    //std::cout<<"\n\ntarget Matrix\n";
    //targetMatrix.printMatrix();
    
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
    
    std::cout<<"\n\noutput Matrix\n";
    int rows = outputLayerMatrix.getrows();
    int cols = outputLayerMatrix.getcols();
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            std::cout<<outputLayerMatrix.get(i,j)<<" ";
        }
        std::cout<<"\n";
    }
    
    float meanlist[chars];
    float tmp;
    std::cout<<"\n\nMean Matrix: \n";
    for (int i = 0; i < rows; i++){
        tmp = 0;
        for (int j = 0; j < cols; j++){
            tmp += outputLayerMatrix.get(i,j);
            //std::cout<<outputLayerMatrix.get(i,j)<<" ";
        }
        tmp = (tmp/classes)*10e4;
        std::cout<<tmp<<"\n";
        meanlist[i] = tmp;
        //std::cout<<"\n";
    }
    std::cout<<"\n\n";
    
    std::cout<<"\n\nSorted Mean Matrix: \n";
    sortMeanList(meanlist, chars);
    
    return 0;        
}

int Trainer::sortMeanList(float* list, int listSize){
    
    char typeOfTrainingChars[] = {  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 
                                    11 ,12 ,13, 14, 15, 16, 17, 18, 19, 20,
                                    21, 22, 23, 24, 25
                                };
    
    
    char caps[] = { 'A','B','C','D','E','F','G','H',//'I',
                    'J','K','L','M','N','O','P','Q','R',
                    'S','T','U','V','W','X','Y','Z'};
    
    targetChars = new char[chars];
    for (int i = 0; i < chars; i++){
        targetChars[i] = caps[typeOfTrainingChars[i]-1];
    } 
    
    /*
    for (int i = 0; i < listSize; i++){ std::cout<<targetChars[i]<<" ";} */
    
    float tmp, differenceTotal = 0;
    char tmpChar;
    for (int i = 0; i < listSize; i++){
        for (int j = 0; j < listSize-i-1; j++){
            if ( list[j] > list[j+1] ){
                tmp = list[j];
                list[j] = list[j+1];
                list[j+1] = tmp;
                
                tmpChar = targetChars[j];
                targetChars[j] = targetChars[j+1];
                targetChars[j+1] = tmpChar;        
            }
        }
    }
    
    rangeData = new float[chars*2];
    rangeChars = new char[chars];
    
    
    std::cout<<"value"<<"  ===  "<<"difference"<<"  ===  "<<"character\n";
    for (int j = 0; j < listSize; j++){
        if (j == 0){
            std::cout<<list[j]<<"  ===  "<< list[j+1]-list[j]<<"  ===  "<< targetChars[j] <<"\n";
            differenceTotal += list[j+1]-list[j];
            
            rangeChars[j] = targetChars[j];
            rangeData[j*2] = std::numeric_limits<float>::min();;
            rangeData[(j*2)+1] = list[j]+(list[j+1]-list[j])/2;
            
        } else if (j < listSize-1) {
            std::cout<<list[j]<<"  ===  "<< list[j+1]-list[j]<<"  ===  "<< targetChars[j] <<"\n";
            differenceTotal += list[j+1]-list[j];
            
            rangeChars[j] = targetChars[j];
            rangeData[j*2] = list[j]-(list[j]-list[j-1])/2;
            rangeData[(j*2)+1] = list[j]+(list[j+1]-list[j])/2;
            
        } else {
            std::cout<<list[j]<<"  ===  "<< 0 <<"  ===  "<< targetChars[j]<< "\n";
            
            rangeChars[j] = targetChars[j];
            rangeData[j*2] = list[j]-(list[j]-list[j-1])/2;
            rangeData[(j*2)+1] = std::numeric_limits<float>::max();
        }
    }
    std::cout<<"\n\n";
    
    differenceMeanList[iterationNo] = differenceTotal/chars;
    iterationNo++;        
    /*
    for(int i = 0; i < chars; i++){
        std::cout<<rangeChars[(i)]<<" ";
        std::cout<<rangeData[(i*2)]<<" ";
        std::cout<<rangeData[(i*2)+1]<<" \n";
    }
    */
    return 0;
}

int Trainer::printdifferenceMeanList(){
    
    std::cout<<"Difference Mean"<<"   ==>   "<<"Iteration"<<"\n";
    for (int j = 0; j < iterationNo; j++){
        std::cout<<differenceMeanList[j]<<"   ==>   "<<(j+1)<<"\n";
    }     
    std::cout<<"\n------------------------------------------\n"<<iterationNo;
}