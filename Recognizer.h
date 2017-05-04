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
    Recognizer(std::string path);
    Recognizer(const Recognizer& orig);
    virtual ~Recognizer();
    
    int initializeImage(std::string imgPath);
    int printThresholdMatrixMatrix();
    
private:
    std::string imgPath;
    Magick::Image img;
    float **imageMatrix, **thresholdMatrix;
    int width, height;
    double range;

};

#endif /* RECOGNIZER_H */

