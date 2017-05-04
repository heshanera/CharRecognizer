/* 
 * File:   Recognizer.cpp
 * Author: heshan
 * 
 * Created on May 4, 2017, 8:50 PM
 */
#include <iostream>
#include <Magick++.h>
#include "Recognizer.h"

Recognizer::Recognizer() { }

Recognizer::Recognizer(const Recognizer& orig) { }

Recognizer::~Recognizer() { }

int Recognizer::recognize(std::string path) {
    
    initializeImage(path);
    printThresholdMatrixMatrix();
    
}

int Recognizer::initializeImage(std::string path)
{
    Magick::InitializeMagick(NULL);
    Magick::Image image(path);
    this->img = image;
    try { 
      
        image.type( Magick::GrayscaleType );
        image.modifyImage();

        int w = image.columns(),h = image.rows();
        int row = 0,column = 0;
        int range = 256; //pow(2, image.modulusDepth());
        
        Magick::PixelPacket *pixels = image.getPixels(0, 0, w, h);

        // creating the pixel matrix
        this->imageMatrix = new float*[h];for(int i = 0; i < h; ++i) this->imageMatrix[i] = new float[w];
        this->thresholdMatrix = new float*[h];for(int i = 0; i < h; ++i) this->thresholdMatrix[i] = new float[w];
        // storing meta data
        this->width = w; this->height = h;
        this->range = range;
        float pixVal;
        
        // creating the inputmatrix
        int size = ( this->width*this->height ) + 1;
        this->inputVector = new int[size];
        inputVector[0] = 2; // adding the bias
        int inVecIndx = 1;
        
        for(row = 0; row < h; row++)
        {
            for(column = 0; column < w; column++)
            {
                Magick::Color color = pixels[w * row + column];
                pixVal = (color.redQuantum()/range)/256;
                this->imageMatrix[row][column] = pixVal;
                
                if ( pixVal > 0.5 ) this->thresholdMatrix[row][column] = 0;
                else this->thresholdMatrix[row][column] = 1;
                
                inputVector[inVecIndx] = thresholdMatrix[row][column];
                inVecIndx++;
                
                //std::cout<< (color.redQuantum()/range)/256 << " ";
            }   
            //std::cout<< std::endl;
        }    
        
    } catch(std::exception &error_ ) { 
        std::cout << "Caught exception: " << error_.what() << std::endl; 
        return 1; 
    }
    return 0; 
    
}

int Recognizer::printThresholdMatrixMatrix(){
    
    for(int rows = 0; rows < this->height; rows++)
    {
        for(int columns = 0; columns < this->width; columns++)    
        {
            std::cout<<thresholdMatrix[rows][columns]<<" ";
        }
        std::cout<<"\n";
    }
    return 0;
}