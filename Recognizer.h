/* 
 * File:   Recognizer.h
 * Author: heshan
 *
 * Created on May 4, 2017, 8:50 PM
 */

#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <Magick++.h>

class Recognizer {
public:
    Recognizer();
    Recognizer(const Recognizer& orig);
    virtual ~Recognizer();
    
    int recognize(std::string path);
    int initializeImage(std::string imgPath);
    int printThresholdMatrixMatrix();
    
private:
    std::string imgPath;
    Magick::Image img;
    float **imageMatrix, **thresholdMatrix;
    int *inputVector;
    int width, height;
    double range;
    
    float** weightMatrix;

};

#endif /* RECOGNIZER_H */

