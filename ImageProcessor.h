/* 
 * File:   ImageProcessor.h
 * Author: heshan
 *
 * Created on May 6, 2017, 6:30 PM
 */

#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <Magick++.h>

class ImageProcessor {
public:
    ImageProcessor();
    ImageProcessor(const ImageProcessor& orig);
    virtual ~ImageProcessor();
    
    int initializeImage(std::string imgPath);
    int createCropedMatrix();
    int* resizeImage();
    int* skeletonize();
    
    int printThresholdMatrix();
    int printCropedMatrix();
    int printResizedMatrix();
    
private:
    std::string imgPath;
    Magick::Image img;
    float **imageMatrix, **thresholdMatrix;
    int width, height;
    double range;
    // boundaries
    int top,bottom,left,right;
    int **croppedMatrix, **resizedMatrix, **skeletonizedMatrix;

};

#endif /* IMAGEPROCESSOR_H */

