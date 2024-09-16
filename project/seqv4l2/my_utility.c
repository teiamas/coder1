// This is necessary for CPU affinity macros in Linux
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <syslog.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <errno.h>

#include <signal.h>

#include "my_utility.h"




/// @brief: A simple function call for our syslog output
/// @param data: Holds a string we are printing onto syslog
/// @param flag: boolean flag to dictate whther to print data or uname -a command
/// @return: None
void syslogPrint(const char *data, const State flag)
{
    char data_t[MAX_STRING_LEN];
    sprintf(data_t, "Massimo Teia: [Course #:%d][Final Project] ", COURSE);
    openlog(data_t, LOG_NDELAY, LOG_DAEMON);

    switch (flag)
    {
    case SYSLOG_UNAME:
    {
        FILE *fp;
        char var[MAX_STRING_LEN];

        // Force syslog to print uname -a with openlog identifier
        // by copying output of uname to char array

        fp = popen("uname -a", "r");
        while (fgets(var, sizeof(var), fp) != NULL)
        {
        };
        pclose(fp);
        syslog(LOG_DEBUG, "%s",  var);
    }
    break;
    case SYSLOG_DATA:

        syslog(LOG_DEBUG,"%s",  data);
        break;
    case SYSLOG_PERROR:

        syslog(LOG_PERROR,"%s",  data);
        break;
    default:
        break;
    }

    closelog();
}



/// @brief get current time in milli-seconds from defined clock type 
/// @brief @define MY_CLOCK_TYPE: contains the clock type to be used
/// @return current time in msec
double getTimeMsec(void)
{
  struct timespec event_ts = {0, 0};

  clock_gettime(MY_CLOCK_TYPE, &event_ts);
  return ((event_ts.tv_sec)*1000.0) + ((event_ts.tv_nsec)/1000000.0);
}


/// @desc: converts the time from a structure to a double precision float in seconds
/// @param tsptr: points to struct timespec containing the time to be converted
/// @return: the time as double in seconds
double realtime   (struct timespec *tsptr)
{
    return ((double)(tsptr->tv_sec) + (((double)tsptr->tv_nsec)/1000000000.0));
}
