#include "npp.h"

#ifdef __cplusplus
extern "C" {
#endif
void reset_CNN(void); //reset to null values
void engage_CNN(int width, int height, Npp8u* d_resizedImage,  char* csv_fname);//load parameters to start
void disengage_CNN();
void runInference();
#ifdef __cplusplus
}
#endif
