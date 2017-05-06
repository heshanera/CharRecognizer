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
    int initializeImage(std::string imgPath);
    int createCropedMatrix();
    int forwardPropagation();
    int resizeImage();
    
    int printThresholdMatrix();
    int printCropedMatrix();
    int printResizedMatrix();
    
    
private:
    std::string imgPath;
    Magick::Image img;
    float **imageMatrix, **thresholdMatrix;
    int *inputVector;
    int width, height;
    double range;
    
    // boundaries
    int top,bottom,left,right;
    int **croppedMatrix, **resizedMatrix;
    
    
    Matrix  InputMatrix, weightMatrix1, weightMatrix2, 
            weightMatrix3, targerMatrix;
    
    
    

};

#endif /* RECOGNIZER_H */

