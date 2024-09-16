#include <time.h>
#include <sys/time.h>

#ifndef SEQV4L2_MY_UTILITY_H_
#define SEQV4L2_MY_UTILITY_H_

//Traceabilit data for the log file entries

#define COURSE 4
#define MAX_STRING_LEN 512

// Of the available user space clocks, CLOCK_MONONTONIC_RAW is typically most precise and not subject to 
// updates from external timer adjustments
//
// However, some POSIX functions like clock_nanosleep can only use adjusted CLOCK_MONOTONIC or CLOCK_REALTIME
//
//#define MY_CLOCK_TYPE CLOCK_REALTIME
//#define MY_CLOCK_TYPE CLOCK_MONOTONIC
#define MY_CLOCK_TYPE CLOCK_MONOTONIC_RAW
//#define MY_CLOCK_TYPE CLOCK_REALTIME_COARSE
//#define MY_CLOCK_TYPE CLOCK_MONTONIC_COARSE

//information we want to be printed in the log file
typedef enum syslogState
{
    SYSLOG_DATA,
    SYSLOG_UNAME,
    SYSLOG_PERROR
} State;


/// @brief: A simple function call for our syslog output
/// @param data: Holds a string we are printing onto syslog
/// @param flag: boolean flag to dictate whther to print data or uname -a command
/// @return: None
extern void   syslogPrint(const char *data, const State flag);

/// @brief get current time in milli-seconds from defined clock type 
/// @brief @define MY_CLOCK_TYPE: contains the clock type to be used
/// @return current time in msec
extern double getTimeMsec(void);


/// @desc: converts the time from a structure to a double precision float in seconds
/// @param tsptr: points to struct timespec containing the time to be converted
/// @return: the time as double in seconds
extern double realtime   (struct timespec *tsptr);

#endif