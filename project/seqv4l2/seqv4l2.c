// Sam Siewert, December 2020
//
// Sequencer Generic Demonstration
//
// Sequencer - 100 Hz 
//                   [gives semaphores to all other services]
// Service_1 - 25 Hz, every 4th Sequencer loop reads a V4L2 video frame
// Service_2 -  1 Hz, every 100th Sequencer loop writes out the current video frame
//
// With the above, priorities by RM policy would be:
//
// Sequencer = RT_MAX	@ 100 Hz
// Servcie_1 = RT_MAX-1	@ 25  Hz
// Service_2 = RT_MIN	@ 1   Hz
//

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

#include "capturelib.h"
#include "gpu_frame_diff.h"
#include "my_utility.h"
#include "NPP_utils.h"
#include "laplacian_filter.h"

#include "image2time_str.h"


#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_MSEC (1000000)
#define NANOSEC_PER_SEC (1000000000)
#define NUM_CPU_CORES (4)
#define TRUE (1)
#define FALSE (0)
// RT: real time, BE best effort
#define RT_CORE (1)
#define BE_CORE (2)

#define NUM_THREADS     (4)
#define NUM_RT_THREADS  (NUM_THREADS-NUM_BE_THREADS)
#define NUM_BE_THREADS  (2)


#define TRACEABILTY_FOR_AUTOGRADE_ON (0)


int abortTest=FALSE;
int abortS1=FALSE, abortS2=FALSE, abortS3=FALSE, abortS4=FALSE;
sem_t semS1, semS2, semS3, semS4;
//struct timespec start_time_val;
double start_realtime;

static timer_t timer_1;
static struct itimerspec itime = {{1,0}, {1,0}};
static struct itimerspec last_itime;

static unsigned long long seqCnt=0;

typedef struct
{
    int threadIdx;
} threadParams_t;


void Sequencer(int id);

void *Service_1_frame_acquisition(void *threadp);
void *Service_2_frame_process(void *threadp);
void *Service_3_frame_storage(void *threadp);
void *Service_4_edge_detection(void *threadp);


void print_scheduler(void);

int v4l2_frame_acquisition_initialization(char *dev_name);
int v4l2_frame_acquisition_shutdown(void);

int v4l2_frame_acquisition_loop(char *dev_name);

void main(int argc, char *argv[])
{
    double current_realtime;

    char *dev_name="/dev/video0";

    int i, rc, scope, flags=0;

    cpu_set_t threadcpu;
    cpu_set_t allcpuset;

    pthread_t threads[NUM_THREADS];
    threadParams_t threadParams[NUM_THREADS];
    pthread_attr_t rt_sched_attr[NUM_THREADS];
    int rt_max_prio, rt_min_prio, cpuidx;

    struct sched_param rt_param[NUM_THREADS];
    struct sched_param main_param;

    pthread_attr_t main_attr;
    pid_t mainpid;

    printf("Initialize GPU\n");
    //printf("Initialize frame diff\n");
    //engage_frame_diff( HRES * VRES );
    printf("Initialize Laplacian\n");
    engage_laplacian(VRES/*pheight*/,HRES/*pwidth*/);

    //initialize the CNN
    printf("Initialize CNN\n");
    reset_CNN();
    engage_CNN(HRES/*width*/,VRES/*height*/, get_d_laplacian_image() /*device_image  buffer*/,
        "out_data.csv" /*csv file name */);

    printf("Frame acquisition initialization\n");
    v4l2_frame_acquisition_initialization(dev_name);

    // required to get camera initialized and ready
    seq_frame_read();

    printf("Starting High Rate Sequencer\n");
    //clock_gettime(MY_CLOCK_TYPE, &); start_realtime=realtime(&start_time_val);
    //printf("START High Rate Sequencer @ sec=%6.9lf\n", (current_realtime - start_realtime));
    //clear syslog
    system("sudo sh -c 'cat /dev/null > /var/log/syslog'");
    // Print out uname on terminal
    syslogPrint(NULL, SYSLOG_UNAME);

   printf("System has %d processors configured and %d available.\n", get_nprocs_conf(), get_nprocs());

   CPU_ZERO(&allcpuset);

   for(i=0; i < NUM_CPU_CORES; i++)
       CPU_SET(i, &allcpuset);

   printf("Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));


    // initialize the sequencer semaphores
    //
    if (sem_init (&semS1, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&semS2, 0, 0)) { printf ("Failed to initialize S2 semaphore\n"); exit (-1); }
    if (sem_init (&semS3, 0, 0)) { printf ("Failed to initialize S3 semaphore\n"); exit (-1); }
    if (sem_init (&semS4, 0, 0)) { printf ("Failed to initialize S4 semaphore\n"); exit (-1); }

    mainpid=getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0){
        perror("main_param");
        exit(-1);
    }
    print_scheduler();


    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);

    // run ALL threads on core RT_CORE
    CPU_ZERO(&threadcpu);
    for(i=0; i < NUM_THREADS; i++)
    {


        if( i < NUM_RT_THREADS){
            cpuidx=(RT_CORE);
            printf("Setting thread %d to core %d\n", i, cpuidx);
            CPU_SET(cpuidx, &threadcpu);
            rc=pthread_attr_init(&rt_sched_attr[i]);
            rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
            rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
            rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);
            
            rt_param[i].sched_priority=rt_max_prio-i;
            pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);
            threadParams[i].threadIdx=i;

        } else {
            cpuidx=(BE_CORE+i-NUM_RT_THREADS);
            printf("Setting thread %d to core %d\n", i, cpuidx);
            CPU_SET(cpuidx, &threadcpu);
            rc=pthread_attr_init(&rt_sched_attr[i]);
            rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
            rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_OTHER);
            rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);
            
            //rt_param[i].sched_priority=rt_max_prio-i;
            pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);
            threadParams[i].threadIdx=i;

        }    
    }

    printf("Service threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));

    // Create Service threads which will block awaiting release for:
    //

    // Servcie_1 = RT_MAX-1	@ 25 Hz
    //
    rt_param[0].sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&rt_sched_attr[0], &rt_param[0]);
    rc=pthread_create(&threads[0],               // pointer to thread descriptor
                      &rt_sched_attr[0],         // use specific attributes
                      //(void *)0,               // default attributes
                      Service_1_frame_acquisition,                 // thread function entry point
                      (void *)&(threadParams[0]) // parameters to pass in
                     );
    if(rc < 0)
        perror("pthread_create for service 1 - V4L2 video frame acquisition");
    else
        printf("pthread_create successful for service 1\n");


    // Service_2 = RT_MAX-2	@ 1 Hz
    //
    rt_param[1].sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&rt_sched_attr[1], &rt_param[1]);
    rc=pthread_create(&threads[1], &rt_sched_attr[1], Service_2_frame_process, (void *)&(threadParams[1]));
    if(rc < 0)
        perror("pthread_create for service 2 - flash frame storage");
    else
        printf("pthread_create successful for service 2\n");


    // Service_3 = RT_MAX-3	@ BE
    //
    //rt_param[2].sched_priority=rt_max_prio-3;
    //pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rt_param[2].sched_priority=0;
    pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rc=pthread_create(&threads[2], &rt_sched_attr[2], Service_3_frame_storage, (void *)&(threadParams[2]));
    if(rc < 0)
        perror("pthread_create for service 3 - flash frame storage");
    else
        printf("pthread_create successful for service 3\n");

    // Service_4 = @ BE
    //
    //rt_param[2].sched_priority=rt_max_prio-3;
    //pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rt_param[3].sched_priority=0;
    pthread_attr_setschedparam(&rt_sched_attr[3], &rt_param[3]);
    rc=pthread_create(&threads[3], &rt_sched_attr[3], Service_4_edge_detection, (void *)&(threadParams[3]));
    if(rc < 0)
        perror("pthread_create for service 4 - laplacian calculation storage");
    else
        printf("pthread_create successful for service 3\n");

    // Wait for service threads to initialize and await relese by sequencer.
    //
    // Note that the sleep is not necessary of RT service threads are created with 
    // correct POSIX SCHED_FIFO priorities compared to non-RT priority of this main
    // program.
    //
    // sleep(1);

    // Create Sequencer thread, which like a cyclic executive, is highest prio    

    printf("Start sequencer\n");

    // Sequencer = RT_MAX	@ 100 Hz
    //
    /* set up to signal SIGALRM if timer expires */
    timer_create(CLOCK_REALTIME, NULL, &timer_1);

    signal(SIGALRM, (void(*)()) Sequencer);


    /* arm the interval timer 1/120Hz*/
    itime.it_interval.tv_sec = 0;
    itime.it_interval.tv_nsec = 8333333;
    itime.it_value.tv_sec = 0;
    itime.it_value.tv_nsec = 8333333;
    //itime.it_interval.tv_sec = 1;
    //itime.it_interval.tv_nsec = 0;
    //itime.it_value.tv_sec = 1;
    //itime.it_value.tv_nsec = 0;

    timer_settime(timer_1, flags, &itime, &last_itime);


    for(i=0;i<NUM_THREADS;i++)
    {
        if(rc=pthread_join(threads[i], NULL) < 0)
		perror("main pthread_join");
	else
		printf("joined thread %d\n", i);
    }
   printf("Frame acquisition shutt down\n");
   v4l2_frame_acquisition_shutdown();
   printf("Disengage laplacian\n");
   disengage_laplacian();
   printf("Disengage CNN\n");
//   disengage_CNN();
//   printf("Disengage frame diff\n");
   disengage_frame_diff();
   printf("\nTEST COMPLETE\n");
}



void Sequencer(int id)
{
    int rc, flags=0;

    // received interval timer signal
    if(abortTest)
    {
        // disable interval timer
        itime.it_interval.tv_sec = 0;
        itime.it_interval.tv_nsec = 0;
        itime.it_value.tv_sec = 0;
        itime.it_value.tv_nsec = 0;
        timer_settime(timer_1, flags, &itime, &last_itime);
	    printf("Disabling sequencer interval timer with abort=%d and %llu\n", abortTest, seqCnt);

	    // shutdown all services
        abortS1=TRUE; abortS2=TRUE; abortS3=TRUE; abortS4=TRUE;
        sem_post(&semS1); sem_post(&semS2); sem_post(&semS3); sem_post(&semS4);

    }
           
    seqCnt++;

    // Release each service at a sub-rate of the generic sequencer rate

    // Service_1 @ 30 Hz
    if((seqCnt % 4) == 0) sem_post(&semS1);

    // Service_2 @ 10 Hz
    if((seqCnt % 12) == 0) sem_post(&semS2);

    // Service_3 @ 1 Hz
    if((seqCnt % 120) == 0) sem_post(&semS3);


    // Service_4 @ 1 Hz 
    if(((seqCnt+60) % 120) == 0) sem_post(&semS4);

}




void *Service_1_frame_acquisition(void *threadp)
{
    struct timespec begin_time_val;
    struct timespec end_time_val;
    double begin_realtime;
    double end_realtime;
    double duration = -1.0;
    double WCET=0.0;
    unsigned long long S1Cnt=0;
    int fcnt=-1; 
    threadParams_t *threadParams = (threadParams_t *)threadp;
    char data[MAX_STRING_LEN];

    // Start up processing and resource initialization
    while(!abortS1) // check for synchronous abort request
    {
	    // wait for service request from the sequencer, a signal handler or ISR in kernel
        sem_wait(&semS1);
	    if(abortS1) break;
            S1Cnt++;
     	// capture the frame from camera and its duration and so update the worst case execution time
        clock_gettime(MY_CLOCK_TYPE, &begin_time_val); begin_realtime=realtime(&begin_time_val);
       	fcnt = seq_frame_read();
        clock_gettime(MY_CLOCK_TYPE, &end_time_val); end_realtime=realtime(&end_time_val);
        /*log duration*/
        if(fcnt > 0 ){
            duration = end_realtime - begin_realtime;
            if( duration > WCET ){ WCET = duration;}
            sprintf(data, "S1: duration=%6.9lf\n",duration); 
            syslogPrint(data, SYSLOG_DATA);                                    
        }
        /*check for abort signal*/
	    if(S1Cnt > 99999) {abortTest=TRUE;};
    }
    //save WCET for offline analysis
    sprintf(data, "S1: WCET=%6.9lf\n",WCET); 
    syslogPrint(data, SYSLOG_DATA);            
    // Resource shutdown here
    //
    pthread_exit((void *)0);
}


void *Service_2_frame_process(void *threadp)
{
    struct timespec begin_time_val;
    struct timespec end_time_val;
    double begin_realtime;
    double end_realtime;
    double duration= -1.0;
    double WCET=0.0;
    double fstart_realtime=0.0;
    unsigned long long S2Cnt=0;
    int process_cnt;
    threadParams_t *threadParams = (threadParams_t *)threadp;
    char data[MAX_STRING_LEN];

    while(!abortS2)
    {
        sem_wait(&semS2);

	    if(abortS2){break;}
        S2Cnt++;

	    // from the capture ring buffer if we have at least 2 we compute the difference between 0 and 1
        // 1 and 2 and if present the third 2 and 3 frames. We search the max and we copy that frame
        // in the write buffer for the storing service (S3) and we free S3's semaphore, to inmform S3 
        // it has a frame to save.
        // In the meantime we calculate the durationWCET for offline timing analysis
        clock_gettime(MY_CLOCK_TYPE, &begin_time_val); begin_realtime=realtime(&begin_time_val);       
	    process_cnt = enque_max_diff_frame();
        // on order of up to milliseconds of latency to get time
        if( process_cnt == 1 ){
            fstart_realtime = realtime(&begin_time_val);
        }
        if( process_cnt > 0 ){
            //ask to save
            //sem_post(&semS3);
            clock_gettime(MY_CLOCK_TYPE, &end_time_val); end_realtime=realtime(&end_time_val);
            duration = end_realtime - begin_realtime;
            if( duration > WCET ){ WCET = duration;}
            /*log duration*/
            sprintf(data, "S2: duration=%6.9lf\n",duration); 
            syslogPrint(data, SYSLOG_DATA);            
        }
    }
    //save WCET for offline analysis
    sprintf(data, "S2: WCET=%6.9lf\n",WCET); 
    syslogPrint(data, SYSLOG_DATA);    
    //thread shutdown
    pthread_exit((void *)0);
}


void *Service_3_frame_storage(void *threadp)
{
    struct timespec begin_time_val;
    struct timespec end_time_val;
    double begin_realtime;
    double end_realtime;
    double duration = -1.0;
    double WCET=0.0;
    unsigned long long S3Cnt=0;
    int store_cnt=0;
    int old_store_cnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;
    char data[MAX_STRING_LEN];
    
    while(!abortS3)
    {
        //waits for frames to be present in the buffer
        sem_wait(&semS3);
           
	    if(abortS3) break;

	    // store the frames present in the buffer and compute the worst case execution time for 
        // offline analysis
        clock_gettime(MY_CLOCK_TYPE, &begin_time_val); begin_realtime=realtime(&begin_time_val);
	    store_cnt = seq_frame_store();
        clock_gettime(MY_CLOCK_TYPE, &end_time_val); end_realtime=realtime(&end_time_val);
        if( store_cnt > 0 ){
            duration = end_realtime - begin_realtime;
            if( duration > WCET ){ WCET = duration;}
            /*log duration*/
            if( store_cnt != old_store_cnt){
                old_store_cnt = store_cnt;
                sprintf(data, "S3: duration=%6.9lf\n",duration); 
                syslogPrint(data, SYSLOG_DATA);
            }
        }
	    // after last write, set synchronous abort
	    if(store_cnt >=  601 /*6001*/) {abortTest=TRUE;};
    }
    //save WCET for offline analysis
    sprintf(data, "S3: WCET=%6.9lf\n",WCET); 
    syslogPrint(data, SYSLOG_DATA);    
    //thread shutdown
    pthread_exit((void *)0);
}



void *Service_4_edge_detection(void *threadp)
{
    struct timespec begin_time_val;
    struct timespec end_time_val;
    double begin_realtime;
    double end_realtime;
    double duration = -1.0;
    double WCET=0.0;
    unsigned long long S3Cnt=0;
    int edge_cnt=0;
    int old_edge_cnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;
    char data[MAX_STRING_LEN];
    
    while(!abortS4)
    {
        //waits for frames to be present in the buffer         
	    if(abortS4) break;
        
        sem_wait(&semS4);
	    // store the frames present in the buffer and compute the worst case execution time for 
        // offline analysis
        clock_gettime(MY_CLOCK_TYPE, &begin_time_val); begin_realtime=realtime(&begin_time_val);
	    edge_cnt = frame_edges_detection();
        clock_gettime(MY_CLOCK_TYPE, &end_time_val); end_realtime=realtime(&end_time_val);
        if( edge_cnt > 0 ){
            duration = end_realtime - begin_realtime;
            if( duration > WCET ){ WCET = duration;}
            /*log duration*/
            if( edge_cnt != old_edge_cnt){
                old_edge_cnt = edge_cnt;
                sprintf(data, "S4: duration=%6.9lf\n",duration); 
                syslogPrint(data, SYSLOG_DATA);
            }
        }
    }
    //save WCET for offline analysis
    //sprintf(data, "S3: WCET=%6.9lf\n",WCET); 
    //syslogPrint(data, SYSLOG_DATA);    
    //thread shutdown
    pthread_exit((void *)0);
}





void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
       case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
       case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER. Exit!\n"); exit(-1);
         break;
       case SCHED_RR:
           printf("Pthread Policy is SCHED_RR. Exit!\n"); exit(-1);
           break;
       //case SCHED_DEADLINE:
       //    printf("Pthread Policy is SCHED_DEADLINE\n"); exit(-1);
       //    break;
       default:
           printf("Pthread Policy is UNKNOWN. Exit!\n"); exit(-1);
   }
}
