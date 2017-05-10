/* 
 * File:   Recognizer.h
 * Author: heshan
 *
 * Created on May 4, 2017, 8:50 PM
 */

#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <Magick++.h>
#include "Matrix.h"

class Recognizer {
public:
    Recognizer();
    Recognizer(const Recognizer& orig);
    virtual ~Recognizer();
    
    int recognize(std::string path);
    int loadWeights();
    int getOutputMatrix();
    int train(int noOfIteration);
    
private:
    std::string imgPath;
    Magick::Image img;
    Matrix inputMatrix, hiddenLayer1Matrix, hiddenLayer2Matrix, outputLayerMatrix;
    Matrix weightMatrix1, weightMatrix2, weightMatrix3;
    
};

#endif /* RECOGNIZER_H */

