#ifndef SEQV4L2_NPP_UTILS_H_
#define SEQV4L2_NPP_UTILS_H_
/// @brief: init ther NPP library
/// @param argc: number of command line parameters
/// @param argv: vector of command line parameters if there is a device number use it
/// @return zero if successfull
extern int npp_init(int argc, char *argv[]);
#endif