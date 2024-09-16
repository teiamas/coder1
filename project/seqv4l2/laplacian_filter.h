
#ifndef _SEQV4L2_LAPLACIAN_FILTER_H_
#define _SEQV4L2_LAPLACIAN_FILTER_H_

/// @brief prepares the data structure before starting the data transfoirmation
/// @brief to be called at startup
/// @param pheight the hight of the image i.e. the number of rows
/// @param pwidth  the widht of the image i.e. the number of columns
void engage_laplacian(int  pheight, int pwidth);

/// @brief frees the data structures when finised to acuire data
/// @param none
void disengage_laplacian(void);

/// @brief converts the yuyv image to gray and calculates a laplacian on it
/// @param h_yuyvImage the original image memorized on the host in yuyv format
/// @param h_laplImage the resulting image after applyng the laplacian filter
void calc_laplacian(
    const unsigned char * const h_yuyvImage ,
          unsigned char * const h_laplImage );

#ifdef __cplusplus
extern "C" {
#endif
    unsigned char* get_d_laplacian_image(void);
#ifdef __cplusplus
}
#endif          
#endif