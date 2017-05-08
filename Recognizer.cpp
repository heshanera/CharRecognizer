/* 
 * File:   Recognizer.cpp
 * Author: heshan
 * 
 * Created on May 4, 2017, 8:50 PM
 */
#include <iostream>
#include <Magick++.h>
#include "Recognizer.h"
#include "ImageProcessor.h"
#include "Trainer.h"
#include "Matrix.h"

Recognizer::Recognizer() { }

Recognizer::Recognizer(const Recognizer& orig) { }

Recognizer::~Recognizer() { }

int Recognizer::recognize(std::string path) {
    
    ImageProcessor imgPrc;
    imgPrc.initializeImage(path);
    imgPrc.createCropedMatrix();
    
    float inputMatrixData[(w*h+1)];
    int* charData;
    charData = imgPrc.resizeImage();
    int brk = 0;
    for (int j = 0; j < (1601); j++) {

        if ( j == 0 ) inputMatrixData[(1601*i)] = 1; // bias
        else inputMatrixData[j + (1601)] = charData[j-1]; 

        if ( j != 0 ) brk++;
        if (j != 0 ) std::cout<<inputMatrixData[j + (1601)]<<" ";
        if (brk%40 == 0) std::cout<<"\n";
    }
    
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
