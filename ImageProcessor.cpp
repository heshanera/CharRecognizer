/* 
 * File:   ImageProcessor.cpp
 * Author: heshan
 * 
 * Created on May 6, 2017, 6:30 PM
 */
#include <iostream>

#include "ImageProcessor.h"

ImageProcessor::ImageProcessor() { }

ImageProcessor::ImageProcessor(const ImageProcessor& orig) { }

ImageProcessor::~ImageProcessor() { }

int ImageProcessor::initializeImage(std::string path)
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
        
        // initializing boundary points
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

int ImageProcessor::createCropedMatrix(){
    
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

int* ImageProcessor::resizeImage(){
        
    int w = right-left;
    int h = bottom-top;
    int* resizedImageMatrix = new int[w*h];
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
    k = 0;
    for(int i = 0; i < 40; i++)
    {
        for(int j = 0; j < 40; j++)
        {
            // filling the image matrix
            Magick::Color color = pixels2[40 * i + j];
            pixVal = (color.redQuantum()/range)/256;
            if ( pixVal > 0.5 ) { this->resizedMatrix[i][j] = 1; resizedImageMatrix[k] = 1; }
            else { this->resizedMatrix[i][j] = 0; resizedImageMatrix[k] = 0; }
            k++;
        } 
    }
    return resizedImageMatrix;
}

int* ImageProcessor::skeletonize(){

    float kernel[8][3][3] =    {   //original kernel 
                                   { { 0, 0, 0},{-1, 1,-1},{ 1, 1, 1} },
                                   { {-1, 0, 0},{ 1, 1, 0},{-1, 1,-1} },
                                   //original kernel 90 degrees rotated
                                   { { 1,-1, 0},{ 1, 1, 0},{ 1,-1, 0} },
                                   { {-1, 1,-1},{ 1, 1, 0},{-1, 0, 0} },
                                   //original kernel 180 degrees rotated
                                   { { 1, 1, 1},{-1, 1,-1},{ 0, 0, 0} },
                                   { {-1, 1,-1},{ 0, 1, 1},{ 0, 0,-1} },
                                   //original kernel 270 degrees rotated
                                   { { 0,-1, 1},{ 0, 1, 1},{ 0,-1, 1} },
                                   { { 0, 0,-1},{ 0, 1, 1},{-1, 1,-1} }
                                };
    
    int w = 40;
    int h = 40;
    int* skeletonizedImageArray = new int[w*h];
    
    int **tmpImage;
    tmpImage = resizedMatrix;
    int kVal1,kVal2,kVal3,kVal4,kVal5,kVal6,kVal7,kVal8,kVal9;
    int k = 0;
    
    
    
    for (int i = 1; i < w-1; i++){
        for (int j = 1; j < h-1; j++){

                for (int k = 0; k < 8; k++){
                //
                // 
                //  kernel Values:  | 1 2 3 |
                //                  | 4 5 6 |
                //                  | 7 8 9 |
                //

                kVal1 = kernel[k][0][0]; kVal2 = kernel[k][0][1]; kVal3 = kernel[k][0][2];
                kVal4 = kernel[k][1][0]; kVal5 = kernel[k][1][1]; kVal6 = kernel[k][1][2];
                kVal7 = kernel[k][2][0]; kVal8 = kernel[k][2][1]; kVal9 = kernel[k][2][2];


                int fit = 1;
                if ( (kVal1 == 1) && (tmpImage[i-1][j-1] != 1)) fit = 0;
                if ( (kVal2 == 1) && (tmpImage[i-1][j] != 1)) fit = 0;
                if ( (kVal3 == 1) && (tmpImage[i-1][j+1] != 1)) fit = 0;
                if ( (kVal4 == 1) && (tmpImage[i][j-1] != 1)) fit = 0;
                if ( (kVal5 == 1) && (tmpImage[i][j] != 1)) fit = 0;
                if ( (kVal6 == 1) && (tmpImage[i][j+1] != 1)) fit = 0;
                if ( (kVal7 == 1) && (tmpImage[i+1][j-1] != 1)) fit = 0;
                if ( (kVal8 == 1) && (tmpImage[i+1][j] != 1)) fit = 0;
                if ( (kVal9 == 1) && (tmpImage[i+1][j+1] != 1)) fit = 0;


                if ( fit == 1 ){

                    // Not operation
                    if ( kVal1 == 0 ) { tmpImage[i-1][j-1] = tmpImage[i-1][j-1]*1; } 
                    else if ( kVal1 == 1 ) { tmpImage[i-1][j-1] = tmpImage[i-1][j-1]*0; }   

                    // Not operation
                    if ( kVal2 == 0 ) { tmpImage[i-1][j] = tmpImage[i-1][j]*1; } 
                    else if ( kVal2 == 1 ) { tmpImage[i-1][j] = tmpImage[i-1][j]*0; } 

                    // Not operation
                    if ( kVal3 == 0 ) { tmpImage[i-1][j+1] = tmpImage[i-1][j+1]*1; } 
                    else if ( kVal3 == 1 ) { tmpImage[i-1][j+1] = tmpImage[i-1][j+1]*0; } 

                    // Not operation
                    if ( kVal4 == 0 ) { tmpImage[i][j-1] = tmpImage[i][j-1]*1; } 
                    else if ( kVal4 == 1 ) { tmpImage[i][j] = tmpImage[i][j-1]*0; } 

                    // Not operation
                    if ( kVal5 == 0 ) { tmpImage[i][j] = tmpImage[i][j]*1; } 
                    else if ( kVal5 == 1 ) { tmpImage[i][j] = tmpImage[i][j]*0; } 

                    // Not operation
                    if ( kVal6 == 0 ) { tmpImage[i][j+1] = tmpImage[i][j+1]*1; } 
                    else if ( kVal6 == 1 ) { tmpImage[i][j+1] = tmpImage[i][j+1]*0; } 

                    // Not operation
                    if ( kVal7 == 0 ) { tmpImage[i+1][j-1] = tmpImage[i+1][j-1]*1; } 
                    else if ( kVal7 == 1 ) { tmpImage[i+1][j-1] = tmpImage[i+1][j-1]*0; } 

                    // Not operation
                    if ( kVal8 == 0 ) { tmpImage[i+1][j] = tmpImage[i+1][j]*1; } 
                    else if ( kVal8 == 1 ) { tmpImage[i+1][j] = tmpImage[i+1][j]*0; } 

                    // Not operation
                    if ( kVal9 == 0 ) { tmpImage[i+1][j+1] = tmpImage[i+1][j+1]*1; } 
                    else if ( kVal9 == 1 ) { tmpImage[i+1][j+1] = tmpImage[i+1][j+1]*0; }           
                }   
            }
        }
        
    }
    
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            std::cout<<tmpImage[i][j]<<" ";          
        }
        std::cout<<"\n";
    }
    std::cout<<"\n\n\n";
    
    return 0;
}

int ImageProcessor::printThresholdMatrix(){
    
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


int ImageProcessor::printCropedMatrix(){
    
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

int ImageProcessor::printResizedMatrix(){
    
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
