/* 
 * File:   Recognizer.cpp
 * Author: heshan
 * 
 * Created on May 4, 2017, 8:50 PM
 */
#include <iostream>
#include <Magick++.h>
#include "Recognizer.h"
#include "Matrix.h"

Recognizer::Recognizer() { }

Recognizer::Recognizer(const Recognizer& orig) { }

Recognizer::~Recognizer() { }

int Recognizer::recognize(std::string path) {
    
    initializeImage(path);
    //printThresholdMatrix();
    createCropedMatrix();
    //printCropedMatrix();
    resizeImage();
    printResizedMatrix();
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
        this->resizedMatrix = new int*[40];for(int i = 0; i < 40; ++i) this->resizedMatrix[i] = new int[40];
        
        // storing meta data
        this->width = w; this->height = h;
        this->range = range;
        float pixVal;
        
        // creating the inputmatrix
        int size = ( this->width*this->height ) + 1;
        this->inputVector = new int[size];
        inputVector[0] = 2; // adding the bias
        int inVecIndx = 1;
        
        this->top = -1;
        this->bottom = -1;
        this->left = -1;
        this->right = -1;
        
        for(row = 0; row < h; row++)
        {
            for(column = 0; column < w; column++)
            {
                // filling the image matrix
                Magick::Color color = pixels[w * row + column];
                pixVal = (color.redQuantum()/range)/256;
                this->imageMatrix[row][column] = pixVal;
                
                // filling the threshold matrix
                if ( pixVal > 0.5 ) this->thresholdMatrix[row][column] = 0;
                else this->thresholdMatrix[row][column] = 1;
                
                // filling the input matrix
                inputVector[inVecIndx] = thresholdMatrix[row][column];
                inVecIndx++;
                
                // finding the boundaries
                if ( thresholdMatrix[row][column] == 1 ){
                    
                    if ( ( top == -1 ) & ( bottom == -1 ) & ( left == -1 ) & ( right == -1 )){
                        top = row; bottom = row; left = column; right = column;
                    }
                    
                    if ( ( left != -1) & ( left > column ) ) {
                            left = column;
                    }
                    
                    if ( ( right != -1) & ( right < column ) ) {
                            right = column;
                    }
                    
                    if ( ( bottom != -1) & ( bottom < row ) ) {
                            bottom = row;
                    }
                }
                
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

int Recognizer::createCropedMatrix(){
    
    // boundaries 
    int w = right-left;
    int h = bottom-top;
    //std::cout<<"top: "<<top<<" bottom: "<<bottom<<" left: "<<left<<" "<<" right: "<<right<<"\n";
    //std::cout<<"w: "<<w<<" h: "<<h<<"\n\n";
    
    // cropped Matrix
    this->croppedMatrix = new int*[h];for(int i = 0; i < h; ++i) this->croppedMatrix[i] = new int[w];
    
    for(int rows = 0; rows < h; rows++)
    {
        for(int columns = 0; columns < w; columns++)    
        {
            this->croppedMatrix[rows][columns] = this->thresholdMatrix[rows+top][columns+left];
        }
    }
    return 0;
}

int Recognizer::resizeImage(){
    
    int w = right-left;
    int h = bottom-top;
    int k = 0;
    
    float *pixels = new float[w*h];
    
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            pixels[k] = croppedMatrix[i][j];
            k++;
        } 
    }
    Magick::Image image( w,h,"R", Magick::FloatPixel, pixels ); 
    
    Magick::Geometry s1  = Magick::Geometry(40, 40);
    s1.aspect(true);
    image.resize(s1);
    
    //image.write("imgs/out.jpg");
    
    ssize_t columns = 40; 
    float pixVal;
    Magick::PixelPacket *pixels2 = image.getPixels(0, 0, 40, 40);
    
    for(int i = 0; i < 40; i++)
    {
        for(int j = 0; j < 40; j++)
        {
            // filling the image matrix
            Magick::Color color = pixels2[40 * i + j];
            pixVal = (color.redQuantum()/range)/256;
            if ( pixVal > 0.5 ) this->resizedMatrix[i][j] = 1;
            else this->resizedMatrix[i][j] = 0;
        } 
    }
    return 0;
}

int Recognizer::forwardPropagation(){
    
    // [ rows x 1 ] X [ 1 x columns]
    // [ rows x 1 ] - size of the input matrix
    // [ [ rows x columns] ] - weight matrix | columns - no of hidden layers
    
     
    return 0;
}



int Recognizer::printThresholdMatrix(){
    
    for(int rows = 0; rows < this->height; rows++)
    {
        for(int columns = 0; columns < this->width; columns++)    
        {
            std::cout<<thresholdMatrix[rows][columns]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n\n";
    return 0;
}


int Recognizer::printCropedMatrix(){
    
    int w = right-left;
    int h = bottom-top;
    for(int rows = 0; rows < h; rows++)
    {
        for(int columns = 0; columns < w; columns++)    
        {
            std::cout<<croppedMatrix[rows][columns]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n\n";
    return 0;
}

int Recognizer::printResizedMatrix(){
    
    int w = 40;
    int h = 40;
    for(int rows = 0; rows < h; rows++)
    {
        for(int columns = 0; columns < w; columns++)    
        {
            std::cout<<resizedMatrix[rows][columns]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n\n";
    return 0;
}