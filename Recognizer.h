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
    int forwardPropagation();
    int train();
    
private:
    std::string imgPath;
    Magick::Image img;
    
};

#endif /* RECOGNIZER_H */

