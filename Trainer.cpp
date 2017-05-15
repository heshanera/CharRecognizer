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
    
    classes = 41; // output node classes ( 25 + 16 )
    chars = 125 + 80; // number of training chars (25*5 + 16*5)
    int w = 40, h = 40; // width x height of a char (in pixels)
    int size = 1600; // width x height
    learningRate = 0.2; // learning rate of the network 
    differenceMeanList = new float[noOfIteration]; //No of training Iterations
    iterationNo = 0;
    
    inputLayerNodes = size + 1;
    hiddenLayer1Nodes = 200;
    hiddenLayer2Nodes = 150;
    
    
    // Initializing the input Matrix **************************************************************/
    std::string trainingImages[] = {"imgs/training/A.jpg","imgs/training/A2.jpg","imgs/training/A3.jpg","imgs/training/A4.jpg","imgs/training/A5.jpg",
                                    "imgs/training/B.jpg","imgs/training/B2.jpg","imgs/training/B3.jpg","imgs/training/B4.jpg","imgs/training/B5.jpg",
                                    "imgs/training/C.jpg","imgs/training/C2.jpg","imgs/training/C3.jpg","imgs/training/C4.jpg","imgs/training/C5.jpg",
                                    "imgs/training/D.jpg","imgs/training/D2.jpg","imgs/training/D3.jpg","imgs/training/D4.jpg","imgs/training/D5.jpg",
                                    "imgs/training/E.jpg","imgs/training/E2.jpg","imgs/training/E3.jpg","imgs/training/E4.jpg","imgs/training/E5.jpg",
                                    "imgs/training/F.jpg","imgs/training/F2.jpg","imgs/training/F3.jpg","imgs/training/F4.jpg","imgs/training/F5.jpg",
                                    "imgs/training/G.jpg","imgs/training/G2.jpg","imgs/training/G3.jpg","imgs/training/G4.jpg","imgs/training/G5.jpg",
                                    "imgs/training/H.jpg","imgs/training/H2.jpg","imgs/training/H3.jpg","imgs/training/H4.jpg","imgs/training/H5.jpg",
                                    //"imgs/training/.jpg","imgs/training/I2.jpg","imgs/training/I3.jpg","imgs/training/I4.jpg","imgs/training/I5.jpg",
                                    "imgs/training/J.jpg","imgs/training/J2.jpg","imgs/training/J3.jpg","imgs/training/J4.jpg","imgs/training/J5.jpg",
                                    "imgs/training/K.jpg","imgs/training/K2.jpg","imgs/training/K3.jpg","imgs/training/K4.jpg","imgs/training/K5.jpg",
                                    "imgs/training/L.jpg","imgs/training/L2.jpg","imgs/training/L3.jpg","imgs/training/L4.jpg","imgs/training/L5.jpg",
                                    "imgs/training/M.jpg","imgs/training/M2.jpg","imgs/training/M3.jpg","imgs/training/M4.jpg","imgs/training/M5.jpg",
                                    "imgs/training/N.jpg","imgs/training/N2.jpg","imgs/training/N3.jpg","imgs/training/N4.jpg","imgs/training/N5.jpg",
                                    "imgs/training/O.jpg","imgs/training/O2.jpg","imgs/training/O3.jpg","imgs/training/O4.jpg","imgs/training/O5.jpg",
                                    "imgs/training/P.jpg","imgs/training/P2.jpg","imgs/training/P3.jpg","imgs/training/P4.jpg","imgs/training/P5.jpg",
                                    "imgs/training/Q.jpg","imgs/training/Q2.jpg","imgs/training/Q3.jpg","imgs/training/Q4.jpg","imgs/training/Q5.jpg",
                                    "imgs/training/R.jpg","imgs/training/R2.jpg","imgs/training/R3.jpg","imgs/training/R4.jpg","imgs/training/R5.jpg",
                                    "imgs/training/S.jpg","imgs/training/S2.jpg","imgs/training/S3.jpg","imgs/training/S4.jpg","imgs/training/S5.jpg",
                                    "imgs/training/T.jpg","imgs/training/T2.jpg","imgs/training/T3.jpg","imgs/training/T4.jpg","imgs/training/T5.jpg",
                                    "imgs/training/U.jpg","imgs/training/U2.jpg","imgs/training/U3.jpg","imgs/training/U4.jpg","imgs/training/U5.jpg",
                                    "imgs/training/V.jpg","imgs/training/V2.jpg","imgs/training/V3.jpg","imgs/training/V4.jpg","imgs/training/V5.jpg",
                                    "imgs/training/W.jpg","imgs/training/W2.jpg","imgs/training/W3.jpg","imgs/training/W4.jpg","imgs/training/W5.jpg",
                                    "imgs/training/X.jpg","imgs/training/X2.jpg","imgs/training/X3.jpg","imgs/training/X4.jpg","imgs/training/X5.jpg",
                                    "imgs/training/Y.jpg","imgs/training/Y2.jpg","imgs/training/Y3.jpg","imgs/training/Y4.jpg","imgs/training/Y5.jpg",
                                    "imgs/training/Z.jpg","imgs/training/Z2.jpg","imgs/training/Z3.jpg","imgs/training/Z4.jpg","imgs/training/Z5.jpg",
    
                                    "imgs/training/a.jpg","imgs/training/a2.jpg","imgs/training/a3.jpg","imgs/training/a4.jpg","imgs/training/a5.jpg",
                                    "imgs/training/b.jpg","imgs/training/b2.jpg","imgs/training/b3.jpg","imgs/training/b4.jpg","imgs/training/b5.jpg",
                                    "imgs/training/d.jpg","imgs/training/d2.jpg","imgs/training/d3.jpg","imgs/training/d4.jpg","imgs/training/d5.jpg",
                                    "imgs/training/e.jpg","imgs/training/e2.jpg","imgs/training/e3.jpg","imgs/training/e4.jpg","imgs/training/e5.jpg",
                                    "imgs/training/f.jpg","imgs/training/f2.jpg","imgs/training/f3.jpg","imgs/training/f4.jpg","imgs/training/f5.jpg",
                                    "imgs/training/g.jpg","imgs/training/g2.jpg","imgs/training/g3.jpg","imgs/training/g4.jpg","imgs/training/g5.jpg",
                                    "imgs/training/h.jpg","imgs/training/h2.jpg","imgs/training/h3.jpg","imgs/training/h4.jpg","imgs/training/h5.jpg",
                                    //"imgs/training/i.jpg","imgs/training/i2.jpg","imgs/training/i3.jpg","imgs/training/i4.jpg","imgs/training/i5.jpg",
                                    "imgs/training/j.jpg","imgs/training/j2.jpg","imgs/training/j3.jpg","imgs/training/j4.jpg","imgs/training/j5.jpg",
                                    "imgs/training/l.jpg","imgs/training/l2.jpg","imgs/training/l3.jpg","imgs/training/l4.jpg","imgs/training/l5.jpg",
                                    "imgs/training/m.jpg","imgs/training/m2.jpg","imgs/training/m3.jpg","imgs/training/m4.jpg","imgs/training/m5.jpg",
                                    "imgs/training/n.jpg","imgs/training/n2.jpg","imgs/training/n3.jpg","imgs/training/n4.jpg","imgs/training/n5.jpg",
                                    "imgs/training/q.jpg","imgs/training/q2.jpg","imgs/training/q3.jpg","imgs/training/q4.jpg","imgs/training/q5.jpg",
                                    "imgs/training/r.jpg","imgs/training/r2.jpg","imgs/training/r3.jpg","imgs/training/r4.jpg","imgs/training/r5.jpg",
                                    "imgs/training/t.jpg","imgs/training/t2.jpg","imgs/training/t3.jpg","imgs/training/t4.jpg","imgs/training/t5.jpg",
                                    "imgs/training/u.jpg","imgs/training/u2.jpg","imgs/training/u3.jpg","imgs/training/u4.jpg","imgs/training/u5.jpg",
                                    "imgs/training/y.jpg","imgs/training/y2.jpg","imgs/training/y3.jpg","imgs/training/y4.jpg","imgs/training/y5.jpg"
                                };
    
    char typeOfTrainingChars[] = {  1, 1, 1, 1, 1,  
                                    2, 2, 2, 2, 2,  
                                    3, 3, 3, 3, 3,  
                                    4, 4, 4, 4, 4,  
                                    5, 5, 5, 5, 5,  
                                    6, 6, 6, 6, 6,  
                                    7, 7, 7, 7, 7,  
                                    8, 8, 8, 8, 8,  
                                    9, 9, 9, 9, 9,  
                                    10,10,10,10,10, 
                                    11,11,11,11,11,
                                    12,12,12,12,12,
                                    13,13,13,13,13, 
                                    14,14,14,14,14, 
                                    15,15,15,15,15, 
                                    16,16,16,16,16, 
                                    17,17,17,17,17, 
                                    18,18,18,18,18, 
                                    19,19,19,19,19, 
                                    20,20,20,20,20,
                                    21,21,21,21,21, 
                                    22,22,22,22,22, 
                                    23,23,23,23,23, 
                                    24,24,24,24,24, 
                                    25,25,25,25,25,
            
                                    26, 26, 26, 26, 26,  
                                    27, 27, 27, 27, 27,  
                                    28, 28, 28, 28, 28,  
                                    29, 29, 29, 29, 29,  
                                    30, 30, 30, 30, 30,  
                                    31, 31, 31, 31, 31,  
                                    32, 32, 32, 32, 32,  
                                    33, 33, 33, 33, 33,  
                                    34, 34, 34, 34, 34,  
                                    35, 35, 35, 35, 35, 
                                    36, 36, 36, 36, 36, 
                                    37, 37, 37, 37, 37, 
                                    38, 38, 38, 38, 38, 
                                    39, 39, 39, 39, 39, 
                                    40, 40, 40, 40, 40, 
                                    41, 41, 41, 41, 41
                                };
    
    
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
    
    float LO = 0.001, HI = 0.0049;
    
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
    
    char typeOfTrainingChars[] = {  1, 1, 1, 1, 1,  
                                    2, 2, 2, 2, 2,  
                                    3, 3, 3, 3, 3,  
                                    4, 4, 4, 4, 4,  
                                    5, 5, 5, 5, 5,  
                                    6, 6, 6, 6, 6,  
                                    7, 7, 7, 7, 7,  
                                    8, 8, 8, 8, 8,  
                                    9, 9, 9, 9, 9,  
                                    10,10,10,10,10, 
                                    11,11,11,11,11,
                                    12,12,12,12,12,
                                    13,13,13,13,13, 
                                    14,14,14,14,14, 
                                    15,15,15,15,15, 
                                    16,16,16,16,16, 
                                    17,17,17,17,17, 
                                    18,18,18,18,18, 
                                    19,19,19,19,19, 
                                    20,20,20,20,20,
                                    21,21,21,21,21, 
                                    22,22,22,22,22, 
                                    23,23,23,23,23, 
                                    24,24,24,24,24, 
                                    25,25,25,25,25
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