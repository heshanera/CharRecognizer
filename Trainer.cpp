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
    
    classes = 3; // output node classes ( 26(uppercase) + 17(lowercase) +10(digits))
    chars = 6; //318; //156 + 102 + 60; // number of training chars (26*6 + 17*6 + 10*6)
    distinctChars = 26;
    w = 40; h = 40; // width x height of a char (in pixels)
    int size = 1600; // width x height
    learningRate = 1.8; //0.017; // learning rate of the network 
    differenceMeanList = new float[noOfIteration]; //No of training Iterations
    iterationNo = 0;
    
    inputLayerNodes = size + 1;
    hiddenLayer1Nodes = 80; //450;
    hiddenLayer2Nodes = 40; //300;
    
    
    // Initializing the input Matrix **************************************************************/
    
    std::string tmpTrainingImages[] = {  
                        
                        "imgs/training/A.png","imgs/training/A2.png","imgs/training/A3.png","imgs/training/A4.png","imgs/training/A5.png","imgs/training/A6.png",
                        "imgs/training/B.png","imgs/training/B2.png","imgs/training/B3.png","imgs/training/B4.png","imgs/training/B5.png","imgs/training/B6.png",
                        "imgs/training/C.png","imgs/training/C2.png","imgs/training/C3.png","imgs/training/C4.png","imgs/training/C5.png","imgs/training/C6.png",
                        "imgs/training/D.png","imgs/training/D2.png","imgs/training/D3.png","imgs/training/D4.png","imgs/training/D5.png","imgs/training/D6.png",
                        "imgs/training/E.png","imgs/training/E2.png","imgs/training/E3.png","imgs/training/E4.png","imgs/training/E5.png","imgs/training/E6.png",
                        "imgs/training/F.png","imgs/training/F2.png","imgs/training/F3.png","imgs/training/F4.png","imgs/training/F5.png","imgs/training/F6.png",
                        "imgs/training/G.png","imgs/training/G2.png","imgs/training/G3.png","imgs/training/G4.png","imgs/training/G5.png","imgs/training/G6.png",
                        "imgs/training/H.png","imgs/training/H2.png","imgs/training/H3.png","imgs/training/H4.png","imgs/training/H5.png","imgs/training/H6.png",
                        "imgs/training/I.png","imgs/training/I2.png","imgs/training/I3.png","imgs/training/I4.png","imgs/training/I5.png","imgs/training/I6.png",
                        "imgs/training/J.png","imgs/training/J2.png","imgs/training/J3.png","imgs/training/J4.png","imgs/training/J5.png","imgs/training/J6.png",
                        "imgs/training/K.png","imgs/training/K2.png","imgs/training/K3.png","imgs/training/K4.png","imgs/training/K5.png","imgs/training/K6.png",
                        "imgs/training/L.png","imgs/training/L2.png","imgs/training/L3.png","imgs/training/L4.png","imgs/training/L5.png","imgs/training/L6.png",
                        "imgs/training/M.png","imgs/training/M2.png","imgs/training/M3.png","imgs/training/M4.png","imgs/training/M5.png","imgs/training/M6.png",
                        "imgs/training/N.png","imgs/training/N2.png","imgs/training/N3.png","imgs/training/N4.png","imgs/training/N5.png","imgs/training/N6.png",
                        "imgs/training/O.png","imgs/training/O2.png","imgs/training/O3.png","imgs/training/O4.png","imgs/training/O5.png","imgs/training/O6.png",
                        "imgs/training/P.png","imgs/training/P2.png","imgs/training/P3.png","imgs/training/P4.png","imgs/training/P5.png","imgs/training/P6.png",
                        "imgs/training/Q.png","imgs/training/Q2.png","imgs/training/Q3.png","imgs/training/Q4.png","imgs/training/Q5.png","imgs/training/Q6.png",
                        "imgs/training/R.png","imgs/training/R2.png","imgs/training/R3.png","imgs/training/R4.png","imgs/training/R5.png","imgs/training/R6.png",
                        "imgs/training/S.png","imgs/training/S2.png","imgs/training/S3.png","imgs/training/S4.png","imgs/training/S5.png","imgs/training/S6.png",
                        "imgs/training/T.png","imgs/training/T2.png","imgs/training/T3.png","imgs/training/T4.png","imgs/training/T5.png","imgs/training/T6.png",
                        "imgs/training/U.png","imgs/training/U2.png","imgs/training/U3.png","imgs/training/U4.png","imgs/training/U5.png","imgs/training/U6.png",
                        "imgs/training/V.png","imgs/training/V2.png","imgs/training/V3.png","imgs/training/V4.png","imgs/training/V5.png","imgs/training/V6.png",
                        "imgs/training/W.png","imgs/training/W2.png","imgs/training/W3.png","imgs/training/W4.png","imgs/training/W5.png","imgs/training/W6.png",
                        "imgs/training/X.png","imgs/training/X2.png","imgs/training/X3.png","imgs/training/X4.png","imgs/training/X5.png","imgs/training/X6.png",
                        "imgs/training/Y.png","imgs/training/Y2.png","imgs/training/Y3.png","imgs/training/Y4.png","imgs/training/Y5.png","imgs/training/Y6.png",
                        "imgs/training/Z.png","imgs/training/Z2.png","imgs/training/Z3.png","imgs/training/Z4.png","imgs/training/Z5.png","imgs/training/Z6.png",

                        "imgs/training/a.png","imgs/training/a2.png","imgs/training/a3.png","imgs/training/a4.png","imgs/training/a5.png","imgs/training/a6.png",
                        "imgs/training/b.png","imgs/training/b2.png","imgs/training/b3.png","imgs/training/b4.png","imgs/training/b5.png","imgs/training/b6.png",
                        "imgs/training/d.png","imgs/training/d2.png","imgs/training/d3.png","imgs/training/d4.png","imgs/training/d5.png","imgs/training/d6.png",
                        "imgs/training/e.png","imgs/training/e2.png","imgs/training/e3.png","imgs/training/e4.png","imgs/training/e5.png","imgs/training/e6.png",
                        "imgs/training/f.png","imgs/training/f2.png","imgs/training/f3.png","imgs/training/f4.png","imgs/training/f5.png","imgs/training/f6.png",
                        "imgs/training/g.png","imgs/training/g2.png","imgs/training/g3.png","imgs/training/g4.png","imgs/training/g5.png","imgs/training/g6.png",
                        "imgs/training/h.png","imgs/training/h2.png","imgs/training/h3.png","imgs/training/h4.png","imgs/training/h5.png","imgs/training/h6.png",
                        "imgs/training/i.png","imgs/training/i2.png","imgs/training/i3.png","imgs/training/i4.png","imgs/training/i5.png","imgs/training/i6.png",
                        "imgs/training/j.png","imgs/training/j2.png","imgs/training/j3.png","imgs/training/j4.png","imgs/training/j5.png","imgs/training/j6.png",
                        "imgs/training/l.png","imgs/training/l2.png","imgs/training/l3.png","imgs/training/l4.png","imgs/training/l5.png","imgs/training/l6.png",
                        "imgs/training/m.png","imgs/training/m2.png","imgs/training/m3.png","imgs/training/m4.png","imgs/training/m5.png","imgs/training/m6.png",
                        "imgs/training/n.png","imgs/training/n2.png","imgs/training/n3.png","imgs/training/n4.png","imgs/training/n5.png","imgs/training/n6.png",
                        "imgs/training/q.png","imgs/training/q2.png","imgs/training/q3.png","imgs/training/q4.png","imgs/training/q5.png","imgs/training/q6.png",
                        "imgs/training/r.png","imgs/training/r2.png","imgs/training/r3.png","imgs/training/r4.png","imgs/training/r5.png","imgs/training/r6.png",
                        "imgs/training/t.png","imgs/training/t2.png","imgs/training/t3.png","imgs/training/t4.png","imgs/training/t5.png","imgs/training/t6.png",
                        "imgs/training/u.png","imgs/training/u2.png","imgs/training/u3.png","imgs/training/u4.png","imgs/training/u5.png","imgs/training/u6.png",
                        "imgs/training/y.png","imgs/training/y2.png","imgs/training/y3.png","imgs/training/y4.png","imgs/training/y5.png","imgs/training/y6.png",

                        "imgs/training/01.png","imgs/training/02.png","imgs/training/03.png","imgs/training/04.png","imgs/training/05.png","imgs/training/06.png",
                        "imgs/training/11.png","imgs/training/12.png","imgs/training/13.png","imgs/training/14.png","imgs/training/15.png","imgs/training/16.png",
                        "imgs/training/21.png","imgs/training/22.png","imgs/training/23.png","imgs/training/24.png","imgs/training/25.png","imgs/training/26.png",
                        "imgs/training/31.png","imgs/training/32.png","imgs/training/33.png","imgs/training/34.png","imgs/training/35.png","imgs/training/36.png",
                        "imgs/training/41.png","imgs/training/42.png","imgs/training/43.png","imgs/training/44.png","imgs/training/45.png","imgs/training/46.png",
                        "imgs/training/51.png","imgs/training/52.png","imgs/training/53.png","imgs/training/54.png","imgs/training/55.png","imgs/training/56.png",
                        "imgs/training/61.png","imgs/training/62.png","imgs/training/63.png","imgs/training/64.png","imgs/training/65.png","imgs/training/66.png",
                        "imgs/training/71.png","imgs/training/72.png","imgs/training/73.png","imgs/training/74.png","imgs/training/75.png","imgs/training/76.png",
                        "imgs/training/81.png","imgs/training/82.png","imgs/training/83.png","imgs/training/84.png","imgs/training/85.png","imgs/training/86.png",
                        "imgs/training/91.png","imgs/training/92.png","imgs/training/93.png","imgs/training/94.png","imgs/training/95.png","imgs/training/96.png",

                    };
    
    char caps[] = { 'A','B','C','D','E','F','G','H','I',
                    'J','K','L','M','N','O','P','Q','R',
                    'S','T','U','V','W','X','Y','Z',
    
                    'a','b','d','e','f','g','h','i','j',
                    'l','m','n','q','r','t','u','y',
    
                    '0','1','2','3','4','5','6','7','8','9',
    
                };
    
    int totalTrainingImages = 318;
    trainingImages = new std::string[totalTrainingImages];
    for(int i = 0; i < totalTrainingImages; i++){
        trainingImages[i] = tmpTrainingImages[i];
    }
    
    int noOfDistinctChars = 53;
    trainedChars = new char[noOfDistinctChars];
    for(int i = 0; i < noOfDistinctChars; i++){
        trainedChars[i] = caps[i];
    }
    return 0;
}

int Trainer::forwardPropagation(){
    
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
    
    return 0;
}

int Trainer::backPropagation(){

    
    // updating weight matrix 3 ( hidden layer 2 --> output layer )
    w3Delta1 = outputLayerMatrix.subtract(targetMatrix);
    w3Delta2 = outputLayerMatrix.hadamardMul(outputLayerMatrix.subtractFrom(1));
    w3Delta3 = hiddenLayer2Matrix.transpose();        
    w3Delta = w3Delta3.matrixMul(w3Delta1.hadamardMul(w3Delta2)).scalarMul(learningRate);   
    w3Delta = w3Delta.scalarDiv(chars);
    
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
    w2Delta = w2Delta.scalarDiv(chars);
    
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
    w1Delta = w1Delta.scalarDiv(chars);
    
    // updating the weights
    weightMatrix3 = weightMatrix3.subtract(w3Delta);
    weightMatrix2 = weightMatrix2.subtract(w2Delta);
    weightMatrix1 = weightMatrix1.subtract(w1Delta);
    //weightMatrix3.printMatrix();
    //weightMatrix2.printMatrix();
    
    return 0;
}

int Trainer::train(int noOfIteration){
    
    initializeWeightMatrices(noOfIteration);
    
    /*********************** opening the weight data file *********************/

    std::ofstream weightData;
    weightData.open ("weights");
    
    weightData <<"inputNodes: "<<inputLayerNodes;
    weightData <<"\nhiddenLayer1Nodes: "<<hiddenLayer1Nodes;
    weightData <<"\nhiddenLayer2Nodes: "<<hiddenLayer2Nodes;
    weightData <<"\noutputNodes: "<<classes;
    weightData <<"\ntrainSet: "<<chars;
    weightData <<"\ndistinctChars: "<<distinctChars;
    weightData <<"\ndistinctCharList: ";
    for(int i = 0; i < distinctChars; i++ ){
        weightData <<trainedChars[i]<<" ";
    }
    weightData <<"\n\n";
    weightData <<"weights: ";
    weightData <<"\n";
    
    /**************************************************************************/
    
    for(int j = 0; j < distinctChars; j++){
        fillMatrixData(j);
        int iterNo = 0;
        for (int i = 0; i < noOfIteration; i++) {
            iterNo++; std::cout<<"Iteration: "<<iterNo<<"\n";
            forwardPropagation();
            backPropagation();
            printOutputLayer();
        }
        
        /**************** writing the current weight matrices *****************/

        weightData <<"\n"<<trainedChars[j]<<"_matrix1: ";
        int rows = weightMatrix1.getrows();
        int cols = weightMatrix1.getcols();
        for (int i = 0; i < rows; i++) {	
            for (int j = 0; j < cols; j++) {	
                float weight = weightMatrix1.get(i,j);
                weightData <<weight<<" ";
            }    
        }	
        weightData <<"\n"<<trainedChars[j]<<"_matrix2: ";
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
        weightData <<"\n"<<trainedChars[j]<<"_matrix3: ";
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
        weightData <<"\n";
        
        /******************************************************************/
        weightMatrix1.~Matrix(); weightMatrix2.~Matrix(); weightMatrix3.~Matrix();
        hiddenLayer1Matrix.~Matrix(); hiddenLayer2Matrix.~Matrix();
        outputLayerMatrix.~Matrix();
        w3Delta.~Matrix(); w3Delta1.~Matrix(); w3Delta2.~Matrix(); w3Delta3.~Matrix();
        w2Delta.~Matrix(); w2Delta1.~Matrix(); w2Delta2.~Matrix(); w2Delta3.~Matrix();
        w1Delta.~Matrix(); w1Delta1.~Matrix(); w1Delta2.~Matrix(); w1Delta3.~Matrix();
        
        
        
    }
}

int Trainer::fillMatrixData(int charNo){
    
    ImageProcessor imgPrc;
    float inputMatrixData[(w*h+1)*chars];
    int *tmpData;
    
    for (int i = 0; i < chars; i++) {
        imgPrc.initializeImage(trainingImages[(6*charNo) + i]);
        imgPrc.createCropedMatrix();
        tmpData = imgPrc.resizeImage();        
      
        int brk = 0;
        for (int j = 0; j < inputLayerNodes; j++) {
            
            if ( j == 0 ) inputMatrixData[(inputLayerNodes*i)] = 1; // bias
            else {
                if (tmpData[j-1] == 1) inputMatrixData[j + (inputLayerNodes*i)] = 1;
                else inputMatrixData[j + (inputLayerNodes*i)] = -1;
            }
                
                
            if ( j != 0 ) brk++;
            //if (j != 0 ) std::cout<<inputMatrixData[j + (1601*i)]<<" ";
            if (j != 0 ) std::cout<<tmpData[j-1]<<" ";
            if (brk%40 == 0) std::cout<<"\n";
        }
        //std::cout<<"\n\n";
    } 
    
    inputMatrix.allocateSize(chars,inputLayerNodes);
    inputMatrix.fillMatrix(inputMatrixData);
    //inputMatrix.printMatrix();
    
    float LO = 0.001;
    float HI = 0.009;
    
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
            /*
            if ( typeOfTrainingChars[i]-1 == j ) targetMatrixData[k] = 1;
            else targetMatrixData[k] = 0;
             */
            
            if ( k < 6 ) targetMatrixData[k] = 1;
            else targetMatrixData[k] = 0;
          
            k++;
        }
   
    }
    targetMatrix.allocateSize(chars,classes);
    targetMatrix.fillMatrix(targetMatrixData);
    targetMatrix.printMatrix();
    
    return 0;
}

int Trainer::writeMetaData(){
    
    std::ofstream weightData;
    weightData.open ("weights");

    weightData <<"inputNodes: "<<inputLayerNodes;
    weightData <<"\nhiddenLayer1Nodes: "<<hiddenLayer1Nodes;
    weightData <<"\nhiddenLayer2Nodes: "<<hiddenLayer2Nodes;
    weightData <<"\noutputNodes: "<<classes;
    weightData <<"\ntrainSet: "<<chars;
    
    weightData <<"\n\n";
    weightData <<"weights: ";
    
    weightData.close();
    
    return 0;
}

int Trainer::writeWeights(){
    
    std::ofstream weightData;
    weightData.open ("weights");
    
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
    
    weightData.close();
    
    return 0;
}

int Trainer::printOutputLayer(){
    
    //std::cout<<"\n\ntarget Matrix\n";
    //targetMatrix.printMatrix();
    
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
    
    std::cout<<"\n\noutput Matrix\n";
    int rows = outputLayerMatrix.getrows();
    int cols = outputLayerMatrix.getcols();
    float tmpW;
    for (int i = 0; i < rows; i++){
        tmpW = 0;
        for (int j = 0; j < cols; j++){
            //std::cout<<floor(outputLayerMatrix.get(i,j) + 0.5)<<" ";
            //std::cout<<outputLayerMatrix.get(i,j)<<" ";
            tmpW +=  outputLayerMatrix.get(i,j);
        }
        tmpW /= classes;
        std::cout<<tmpW<<" ";
        std::cout<<"\n";
    }

    return 0;        
}