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
    //imgPrc.printThresholdMatrix();
    imgPrc.createCropedMatrix();
    //imgPrc.printCropedMatrix();
    imgPrc.resizeImage();
    imgPrc.printResizedMatrix();
}

int Recognizer::forwardPropagation(){
    
    
    
    // [ rows x 1 ] X [ 1 x columns]
    // [ rows x 1 ] - size of the input matrix
    // [ [ rows x columns] ] - weight matrix | columns - no of hidden layers
    
     
    return 0;
}

int Recognizer::train(){

    Trainer trainer;
    trainer.initializeWeightMatrices();
    trainer.forwardPropagation();
    trainer.backPropagation();
    
}
