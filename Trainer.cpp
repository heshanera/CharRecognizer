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
    
    int classes = 26;
    int chars = 8;
    int w = 40, h = 40;
    int size = 1600;
    
    
    // Initializing the input Matrix **************************************************************/
    std::string trainingImages[] = {"imgs/training/A.jpg","imgs/training/B.jpg","imgs/training/C.jpg",
                                    "imgs/training/D.jpg","imgs/training/E.jpg","imgs/training/F.jpg",
                                    "imgs/training/G.jpg","imgs/training/H.jpg"};
    ImageProcessor imgPrc;
    int inputMatrixData[(w*h+1)*chars];
    int* tmpData;
    
    for (int i = 0; i < chars; i++) {
        
        imgPrc.initializeImage(trainingImages[i]);
        imgPrc.createCropedMatrix();
        tmpData = imgPrc.resizeImage();
        
        inputMatrixData[(1600*i)] = 2; // bias
        for (int j = 1; j < (1601); j++) {
            inputMatrixData[j + (1600*i)] = tmpData[j];
            //std::cout<<inputMatrixData[j+ (1600*i)]<<" ";
            //if (j%40 == 0) std::cout<<"\n";
        }
        //std::cout<<"\n\n";
    }    
    inputMatrix.allocateSize(chars,1601 /* = width x height + bias = 40*40+1 */);
    
    // Initializing the weight Matrix1 **************************************************************/
    int hiddenLayer1Nodes = 500;
    int charPixSize = 1601 /* = width x height + bias = 40*40+1 */;
    float randomFloat;
    float weightMatrix1Data[ charPixSize * hiddenLayer1Nodes];
    int k = 0;
    for(int i = 0; i < charPixSize; i++){
        for(int j = 0; j < hiddenLayer1Nodes; j++){
            randomFloat = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
            weightMatrix1Data[k] = randomFloat;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix1.allocateSize(charPixSize, hiddenLayer1Nodes);
    weightMatrix1.fillMatrix(weightMatrix1Data);
    
    // Initializing the weight Matrix2 **************************************************************/
    int hiddenLayer2Nodes = 750;
    float weightMatrix2Data[ hiddenLayer1Nodes * hiddenLayer2Nodes];
    k = 0;
    for(int i = 0; i < hiddenLayer1Nodes; i++){
        for(int j = 0; j < hiddenLayer2Nodes; j++){
            randomFloat = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
            weightMatrix2Data[k] = randomFloat;
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
            randomFloat = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
            weightMatrix3Data[k] = randomFloat;
            //std::cout<<randomFloat<<" " ;
        }
    }
    weightMatrix3.allocateSize(hiddenLayer2Nodes,outputLayerNodes);
    weightMatrix3.fillMatrix(weightMatrix3Data);
    
    // Initializing the target Matrix **************************************************************/
    float targetMatrixData[classes*chars];
    k = 0;
    for (int i = 0; i < classes; i++){
        for (int j = 0; j < chars; j++){
            if ( i == j ) targetMatrixData[k] = 1;
            else targetMatrixData[k] = 0;
        }
   
    }
    targetMatrix.allocateSize(classes,chars);
    targetMatrix.fillMatrix(targetMatrixData);

}
