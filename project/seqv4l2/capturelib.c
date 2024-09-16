/*
 *
 *  Adapted by Sam Siewert for use with UVC web cameras and Bt878 frame
 *  grabber NTSC cameras to acquire digital video from a source,
 *  time-stamp each frame acquired, save to a PGM or PPM file.
 *
 *  The original code adapted was open source from V4L2 API and had the
 *  following use and incorporation policy:
 * 
 *  This program can be used and distributed without restrictions.
 *
 *      This program is provided with the V4L2 API
 * see http://linuxtv.org/docs.php for more information
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <syslog.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#include <time.h>
#include "ring_buffer.h"
#include "my_utility.h"

#include <pthread.h>

#include "gpu_frame_diff.h"
#include "laplacian_filter.h"
#include "capturelib.h"
#include "image2time_str.h"

// --- extern imports ---

extern double start_realtime; //frerunning timer value when we started the frames capturing used to have a relative time

//utilities macroes
#define CLEAR(x) memset(&(x), 0, sizeof(x))

// camera data definition macroes
#define MAX_HRES (1920)
#define MAX_VRES (1080)
#define MAX_PIXEL_SIZE (3)


#define PIXEL_SIZE (2)
#define HRES_STR "640"
#define VRES_STR "480"

//#define HRES (320)
//#define VRES (240)
//#define PIXEL_SIZE (2)
//#define HRES_STR "320"
//#define VRES_STR "240"

#define FRAME_SIZE (HRES*VRES*PIXEL_SIZE)

#define STARTUP_FRAMES (150)
#define LAST_FRAMES (1)

//#define FRAMES_PER_SEC (1) 
//#define FRAMES_PER_SEC (5)
//#define FRAMES_PER_SEC (10) 
//#define FRAMES_PER_SEC (20) 
//#define FRAMES_PER_SEC (25) 
#define FRAMES_PER_SEC (30) 

#define FRAMES_PER_CAPTURE (3)
#define MIN_FRAMES_PER_CAPTURE (2)
#define COLOR_CONVERT_RGB

#define READ_RING_BUFFER_SIZE  ( 30)
#define WRITE_RING_BUFFER_SIZE ( 90)
#define EDGE_RING_BUFFER_SIZE  ( 90)
//#define COLOR_CONVERT_GRAY
#define DUMP_FRAMES

#define DRIVER_MMAP_BUFFERS (6)  // request buffers for delay


#define TRACE_STORE_SERVICE (0)    

// data buffer type
struct buffer 
{
        void   *start;
        size_t  length;
};

// Format is used by a number of functions, so made as a file global
static struct v4l2_format fmt;
struct v4l2_buffer frame_buf;


static  struct ring_buffer_t	ring_buffer;        //we store here the captured images waiting to be 
                                                    // searched to find the one containing the tick
static  struct ring_buffer_t	write_ring_buffer;  // we store here the tick images waiting to be saved
static  struct ring_buffer_t	edge_ring_buffer;   // we store here the images waiting to be processed
                                                    // for edge highlight and saved
//camera data
static int              camera_device_fd = -1;
struct buffer          *buffers;
static unsigned int     n_buffers;
static int              force_format=1;

// time of start acquisition and stop acquisition as floating in sec and as strutures with 
// field for sec and nano sec.
static double fstart=0.0, fstop=0.0;
static struct timespec time_now, time_start, time_stop;

// global counters for frames
// always ignore STARTUP_FRAMES while camera adjusts to lighting, focuses, etc.
int read_framecnt    = -STARTUP_FRAMES;
int process_framecnt = 0;
int save_framecnt    = 0;
int edge_framecnt    = 0;

// buffers used for temporary conversions
unsigned char scratchpad_buffer[MAX_HRES*MAX_VRES*MAX_PIXEL_SIZE];
unsigned char edge_scratchpad_buffer[MAX_HRES*MAX_VRES*MAX_PIXEL_SIZE];
unsigned char edge_scratchpad_buffer_out[MAX_HRES*MAX_VRES*MAX_PIXEL_SIZE];

static void errno_exit(const char *s)
{
        fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
        exit(EXIT_FAILURE);
}

/// @brief execute an io control request managing errors
/// @param fh damera device file descriptor 
/// @param request command to be executed
/// @param arg arguments of the command
/// @return command result
static int xioctl(int fh, int request, void *arg)
{
    int rc;

    do 
    {
        rc = ioctl(fh, request, arg);

    } while (-1 == rc && EINTR == errno);

    return rc;
}

/// @brief headers templates for the ppm header record and ppm file names
char ppm_header[]="P6\n#9999999999 sec 9999999999 msec \n"HRES_STR" "VRES_STR"\n255\n";
char ppm_dumpname[]="frames/test00000.ppm";

/// @brief save a yuyv image to a ppm file
/// @param p frame data buffer pointer
/// @param size number of bytes of the frame
/// @param tag tag for the file name of this frame
/// @param time time when this frame was acquired
static void dump_ppm(const void *p, int size, unsigned int tag, struct timespec *time)
{
    int written, i, total, dumpfd;
   
    snprintf(&ppm_dumpname[11], 10, "%05d.ppm", tag);
    dumpfd = open(ppm_dumpname, O_WRONLY | O_NONBLOCK | O_CREAT | O_SYNC, 00666);
    if (dumpfd < 0) {
        char err_mess[1024];
        snprintf(err_mess, sizeof(err_mess), "Error on dump file opening: %s\n", ppm_dumpname);
        errno_exit(err_mess);
    }
    snprintf(ppm_header, sizeof(ppm_header), "P6\n#%010d sec %010d msec \n"HRES_STR" "VRES_STR"\n255\n",
             (int)time->tv_sec, (int)((time->tv_nsec) / 1000000));
    
    written = write(dumpfd, ppm_header, strlen(ppm_header));
    if (written < 0) {
        close(dumpfd);
        errno_exit("Error writing ppm header");
    }

    total = 0;
    while (total < size) {
        written = write(dumpfd, p + total, size - total);
        if (written < 0) {
            close(dumpfd);
            errno_exit("Error writing ppm data");
        }
        total += written;
    }

    close(dumpfd);
}


/// @brief headerrs templates for the ppm header record and pgm file names
char pgm_header[]="P5\n#9999999999 sec 9999999999 msec \n"HRES_STR" "VRES_STR"\n255\n";
char pgm_dumpname[]="frames/test0000.pgm";

/// @brief save a gray image to a pgm file
/// @param p frame data buffer pointer
/// @param size number of bytes of the frame
/// @param tag tag for the file name of this frame
/// @param time time when this frame was acquired
static void dump_pgm(const void *p, int size, unsigned int tag, struct timespec *time)
{
    int written, i, total, dumpfd;
   
    snprintf(&pgm_dumpname[11], 9, "%04d", tag);
    strncat(&pgm_dumpname[15], ".pgm", 5);
    dumpfd = open(pgm_dumpname, O_WRONLY | O_NONBLOCK | O_CREAT, 00666);

    snprintf(&pgm_header[4], 11, "%010d", (int)time->tv_sec);
    strncat(&pgm_header[14], " sec ", 5);
    snprintf(&pgm_header[19], 11, "%010d", (int)((time->tv_nsec)/1000000));
    strncat(&pgm_header[29], " msec \n"HRES_STR" "VRES_STR"\n255\n", 19);

    // subtract 1 from sizeof header because it includes the null terminator for the string
    written=write(dumpfd, pgm_header, sizeof(pgm_header)-1);

    total=0;

    do
    {
        written=write(dumpfd, p, size);
        total+=written;
    } while(total < size);

    clock_gettime(CLOCK_MONOTONIC, &time_now);

    close(dumpfd);
    
}


/// @brief convert from yuyv (y is the luminance gamma corrected, u v are the chromatic part) data format from camera 
///        to gray taking only  the luma channel data
/// @param yuyv_data_ptr pointer to original data buffer
/// @param gray_data_ptr pointer to transformed data buffer
/// @param size size of the buffer
void convert_yuyv2gray(const unsigned char* const yuyv_data_ptr,unsigned char* const gray_data_ptr, int size){
    // Pixels are YU and YV alternating, so YUYV which is 4 bytes
    // We want Y, so YY which is 2 bytes
    //    
    for(int i=0, newi=0; i<size; i=i+4, newi=newi+2)
    {
        // Y1=first byte and Y2=third byte
        gray_data_ptr[newi  ] = yuyv_data_ptr[i  ];
        gray_data_ptr[newi+1] = yuyv_data_ptr[i+2];
    }
}

/// @brief takes data from edge ring buffer, transforms the image applying the Laplacian filter
/// @brief and so highlighting the edges and  save it
/// @param none
/// @return the number of images processed
int frame_edges_detection(void)
{
    int frames_num=edge_ring_buffer.count;
    /*buffer for log*/
    char data[MAX_STRING_LEN];
    //MST <<< unsigned char *edged_data_ptr=(unsigned char*) 0;
    for(int frame_cnt= frames_num; frame_cnt>0; frame_cnt--){

        clock_gettime(CLOCK_MONOTONIC, &time_now);       
        // convert the image to rgb and save result in the scratch pad
        calc_laplacian(
           &((edge_ring_buffer.save_frame[edge_ring_buffer.head_idx].frame[0])),/*h_yuyvImage,*/
           edge_scratchpad_buffer /*h_laplImage,*/
        );
        /*new frame elaborated increment the counter*/
        edge_framecnt++;
        // recognize the time and save as csv
        //runInference(Npp8u* pDst, int width, int height, const char* csv_file)
        //MST ===>
        runInference();
        //MST<===
        dump_pgm(edge_scratchpad_buffer, (FRAME_SIZE/2), edge_framecnt, &time_now);
        // frame processed  free the write ring buffer positions
        rb_remove(&edge_ring_buffer,1);
        // save log message for debug
        sprintf(data, "S4: Edge            :   edge_framecnt=%5.5d, erb_head_idx=%2.2d, erb_tail=%2.2d, erb_count=%2.2d, time=%014.9f\n",
                edge_framecnt, edge_ring_buffer.head_idx, edge_ring_buffer.tail_idx, edge_ring_buffer.count,
                realtime(&time_now) -start_realtime );
        syslogPrint(data, SYSLOG_DATA);    
        
    }

    return edge_framecnt;
}
    

/// @brief convert the yuv image to rgb
/// @param y luma channel (gamma corrected luminance)
/// @param u u chroma chnnel
/// @param v v chroma channel 
/// @param r red channel data
/// @param g green channel data
/// @param b blu channel data
void yuv2rgb_float(float y, float u, float v, 
                   unsigned char *r, unsigned char *g, unsigned char *b)
{
    float r_temp, g_temp, b_temp;

    // R = 1.164(Y-16) + 1.1596(V-128)
    r_temp = 1.164*(y-16.0) + 1.1596*(v-128.0);  
    *r = r_temp > 255.0 ? 255 : (r_temp < 0.0 ? 0 : (unsigned char)r_temp);

    // G = 1.164(Y-16) - 0.813*(V-128) - 0.391*(U-128)
    g_temp = 1.164*(y-16.0) - 0.813*(v-128.0) - 0.391*(u-128.0);
    *g = g_temp > 255.0 ? 255 : (g_temp < 0.0 ? 0 : (unsigned char)g_temp);

    // B = 1.164*(Y-16) + 2.018*(U-128)
    b_temp = 1.164*(y-16.0) + 2.018*(u-128.0);
    *b = b_temp > 255.0 ? 255 : (b_temp < 0.0 ? 0 : (unsigned char)b_temp);
}


// This is probably the most acceptable conversion from camera YUYV to RGB
//
// Wikipedia has a good discussion on the details of various conversions and cites good references:
// http://en.wikipedia.org/wiki/YUV
//
// Also http://www.fourcc.org/yuv.php
//
// What's not clear without knowing more about the camera in question is how often U & V are sampled compared
// to Y.
//
// E.g. YUV444, which is equivalent to RGB, where both require 3 bytes for each pixel
//      YUV422, which we assume here, where there are 2 bytes for each pixel, with two Y samples for one U & V,
//              or as the name implies, 4Y and 2 UV pairs
//      YUV420, where for every 4 Ys, there is a single UV pair, 1.5 bytes for each pixel or 36 bytes for 24 pixels

void yuv2rgb(int y, int u, int v, unsigned char *r, unsigned char *g, unsigned char *b)
{
   int r1, g1, b1;

   // replaces floating point coefficients
   int c = y-16, d = u - 128, e = v - 128;       

   // Conversion that avoids floating point
   r1 = (298 * c           + 409 * e + 128) >> 8;
   g1 = (298 * c - 100 * d - 208 * e + 128) >> 8;
   b1 = (298 * c + 516 * d           + 128) >> 8;

   // Computed values may need clipping.
   if (r1 > 255) r1 = 255;
   if (g1 > 255) g1 = 255;
   if (b1 > 255) b1 = 255;

   if (r1 < 0) r1 = 0;
   if (g1 < 0) g1 = 0;
   if (b1 < 0) b1 = 0;

   *r = r1 ;
   *g = g1 ;
   *b = b1 ;
}



/// @brief save an image from the camera format to the the one requested by the macroes definition
/// @param p frame data pointer
/// @param size frame size in bytes
/// @param frame_time acuisition time of the frame
static void  save_image(const void *p, int size, struct timespec *frame_time)
{
    int i, newi, newsize=0;
    unsigned char *frame_ptr = (unsigned char *)p;

    save_framecnt++;
    
#ifdef DUMP_FRAMES	

    if(fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_GREY)
    {
        dump_pgm(frame_ptr, size, save_framecnt, frame_time);
    }

    else if(fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV)
    {

        #if defined(COLOR_CONVERT_RGB)
           
            if(save_framecnt > 0 && frame_ptr != (unsigned char *)0) 
            {
                dump_ppm(frame_ptr, ((size*6)/4), save_framecnt, frame_time);
            }
        #elif defined(COLOR_CONVERT_GRAY)
            if(save_framecnt > 0)
            {
                dump_pgm(frame_ptr, (size/2), process_framecnt, frame_time);
            }
        #endif

    }

    else if(fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_RGB24)
    {
        dump_ppm(frame_ptr, size, process_framecnt, frame_time);
    }
    else
    {
        printf("ERROR - unknown dump format\n");
    }
#endif

}

/// @brief convert a camera image to the one requested by the macroes
/// @param p frame data pointewr
/// @param size frame size in Bytes
/// @return number of processed images
static int process_image(const void *p, int size)
{
    int i, newi, newsize=0;
    int y_temp, y2_temp, u_temp, v_temp;
    unsigned char *frame_ptr = (unsigned char *)p;

    #ifdef PRINTF_NO_RT    
        printf("process frame %d: ", process_framecnt);
    #endif
    if(fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_GREY)
    {
        printf("NO PROCESSING for graymap as-is size %d\n", size);
    }

    else if(fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV)
    {
#if defined(COLOR_CONVERT_RGB)
        #ifdef PRINTF_NO_RT
            printf("PROCESSING convert to rgb, size=%d\n", size);
        #endif
        // Pixels are YU and YV alternating, so YUYV which is 4 bytes
        // We want RGB, so RGBRGB which is 6 bytes
        //
        for(i=0, newi=0; i<size; i=i+4, newi=newi+6)
        {
            y_temp=(int)frame_ptr[i]; u_temp=(int)frame_ptr[i+1]; y2_temp=(int)frame_ptr[i+2]; v_temp=(int)frame_ptr[i+3];
            yuv2rgb(y_temp, u_temp, v_temp, &scratchpad_buffer[newi], &scratchpad_buffer[newi+1], &scratchpad_buffer[newi+2]);
            yuv2rgb(y2_temp, u_temp, v_temp, &scratchpad_buffer[newi+3], &scratchpad_buffer[newi+4], &scratchpad_buffer[newi+5]);
        }
#elif defined(COLOR_CONVERT_GRAY)
        // Pixels are YU and YV alternating, so YUYV which is 4 bytes
        // We want Y, so YY which is 2 bytes
        //
        for(i=0, newi=0; i<size; i=i+4, newi=newi+2)
        {
            // Y1=first byte and Y2=third byte
            scratchpad_buffer[newi]=frame_ptr[i];
            scratchpad_buffer[newi+1]=frame_ptr[i+2];
        }
#endif
    }

    else if(fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_RGB24)
    {
        printf("NO PROCESSING for RGB as-is size %d\n", size);
    }
    else
    {
        printf("NO PROCESSING ERROR - unknown format\n");
    }

    return process_framecnt;
}

/// @brief read a frame from the camer: dequeue it from kernel buffers
/// @param  globally defined camera buffer parameters
/// @return counter of frames read
static int read_frame(void)
{
    CLEAR(frame_buf);

    frame_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    frame_buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(camera_device_fd, VIDIOC_DQBUF, &frame_buf))
    {
        switch (errno)
        {
            case EAGAIN:
                return 0;

            case EIO:
                /* Could ignore EIO, but drivers should only set for serious errors, although some set for
                   non-fatal errors too.
                 */
                return 0;


            default:
                printf("mmap failure\n");
                errno_exit("VIDIOC_DQBUF");
        }
    }

    read_framecnt++;

    //printf("frame %d ", read_framecnt);

    assert(frame_buf.index < n_buffers);

    return 1;
}

/// @brief dequeue the read frame from kernel buffers insert in the ring buffer and start next frame read
/// @param  none
/// @return read frames counter
int seq_frame_read(void)
{

    fd_set fds;
    struct timeval tv;
    int rc;

    //set file descriptor point to camera
    FD_ZERO(&fds);
    FD_SET(camera_device_fd, &fds);

    /* Timeout */
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    /*buffer for log*/
    char data[MAX_STRING_LEN];

    rc = select(camera_device_fd + 1, &fds, NULL, NULL, &tv);
    //save time when camera reads frame
    clock_gettime(CLOCK_MONOTONIC, &time_now);
    read_frame();
    // first frame save the time when acuisition started
    if( read_framecnt >= 1){ 
        //start to insert the frames in the Ring Buffer
        if( read_framecnt == 1) 
        {
            //if frame is zero this is the start of frame acquistions so save it
            time_start = time_now;
            fstart = realtime(&time_start);
        } 
        //insert frame in the queue
        rb_insert(&ring_buffer, 
                buffers[frame_buf.index].start, 
                frame_buf.bytesused,
                read_framecnt,
                &time_now);
        // save log message for debug
        sprintf(data, "S1: Insert           :    read_framecnt=%5.5d, rrb_head_idx=%2.2d, rrb_tail=%2.2d, rrb_count=%2.2d, time=%014.9f\n",
                read_framecnt, ring_buffer.head_idx, ring_buffer.tail_idx, ring_buffer.count,
                realtime(&time_now) -start_realtime );
        syslogPrint(data, SYSLOG_DATA);    
        //queue next frame
    }
    if (-1 == xioctl(camera_device_fd, VIDIOC_QBUF, &frame_buf))
        errno_exit("VIDIOC_QBUF");
    return read_framecnt;
}
#if 1
unsigned int diff_image(int br_index,int size,char* const pptr,char* const prev_pptr){
    // this function calculates the difference beteen the two images
    unsigned int diff=0;
    for(int i=0; i<size; i++){
        diff += abs(pptr[i]-prev_pptr[i]);
    }
    return diff;
}
#endif


struct save_frame_t* search_max_frame_diff(unsigned int size ){
    // This function searches the max difference in the last elements of the ring buffer
    // input: global ring_buffer
    // output: pointer to the max difference buffer
    char data[MAX_STRING_LEN];

    //check how many elements we have in the buffer to set the number of elements to search
    int elem_num = 0;
    if( ring_buffer.count >= FRAMES_PER_CAPTURE ){
        elem_num = FRAMES_PER_CAPTURE;
    } else {
        sprintf(data, "Search Max diff. error: element num < 3: %d",ring_buffer.count);
        syslogPrint(data, SYSLOG_DATA);
    }

    //loop trought the last elements compute the max difference max(frame(n)-frame(n-1))
    //initialize to current head
    struct save_frame_t* curr_frame_ptr = (struct save_frame_t*)0;
    struct save_frame_t* prev_frame_ptr = (struct save_frame_t*)0;
    //points to the frame with the max difference in the last FRAMES_PER_CAPTURE
    struct save_frame_t* max_diff_frame_ptr = (struct save_frame_t*)0;

    unsigned int max_diff  = 0;
    unsigned int curr_diff = 0;
    //at the beginning all point to first element
    prev_frame_ptr = max_diff_frame_ptr = curr_frame_ptr = &(ring_buffer.save_frame[ ring_buffer.head_idx ]);        
    //loop starting from one not zero
    for(int i=1, ring_index=0; i<elem_num; i++)
    { 
        //update current module ring_size
        ring_index = (ring_buffer.head_idx + i)% ring_buffer.ring_size;
        curr_frame_ptr = ring_buffer.save_frame + ring_index;
        //curr_diff = do_frame_diff(curr_frame_ptr->frame, prev_frame_ptr->frame);
        curr_diff = diff_image(ring_index, size, curr_frame_ptr->frame, prev_frame_ptr->frame);
        //syslog(LOG_CRIT, "S2 Diff new_fcnt=%s olf_fcnt=%s br.index=%d, Diff_Value=%d\n",
        //                  curr_frame_ptr->identifier_str,
        //                  prev_frame_ptr->identifier_str,
        //                   ring_index, curr_diff);
        if( curr_diff > max_diff ){
            max_diff_frame_ptr = curr_frame_ptr;
            max_diff = curr_diff;
            double maxf_time_stamp = (double)max_diff_frame_ptr->time_stamp.tv_sec + 
                                    (double)max_diff_frame_ptr->time_stamp.tv_nsec / 1000000000.0;
            //syslog(LOG_CRIT, "S2 new MaxDiffFrame  frameCnt=%s Diff=%u MaxDiff=%u at=%lf"  , 
            //                  max_diff_frame_ptr->identifier_str, 
            //                                        curr_diff, max_diff, 
            //                                                 maxf_time_stamp-fstart);
        }
        prev_frame_ptr = curr_frame_ptr;
    }
    //return the pointer to the difference that has the max 
    return max_diff_frame_ptr;
}

/// @brief search the frame with the maximum difference between itslef and the one before it
/// @brief when found enqueue it
/// @param  none
/// @return read frames counter
int enque_max_diff_frame(void)
{
    //points to the frame with the max difference in the last FRAMES_PER_CAPTURE
    struct save_frame_t* max_diff_frame_ptr = (struct save_frame_t*)0;
    const int frame_size=FRAME_SIZE;
    char data[MAX_STRING_LEN];
    
    if( read_framecnt > 0 ) {
        if( ring_buffer.count >= FRAMES_PER_CAPTURE  ){
            while( ring_buffer.count >= FRAMES_PER_CAPTURE  ){
                process_framecnt++;        //compute the difference between this image and the previous one
                //printf("pfcnt: %d\n",process_framecnt);
                max_diff_frame_ptr = search_max_frame_diff(frame_size);
                if( process_framecnt > 0 ) {
                    //insert it in the write buffer waiting to be saved
                    //copy data
                    clock_gettime(CLOCK_MONOTONIC, &time_now);
                    rb_insert(&write_ring_buffer, max_diff_frame_ptr->frame, frame_size,read_framecnt,
                            &time_now);
                    rb_insert(&edge_ring_buffer, max_diff_frame_ptr->frame, frame_size,read_framecnt,
                            &time_now);
                    // save log data for debug                             
                    sprintf(data, "S2: Write Ring buffer: process_framecnt=%5.5d, wrb_head_idx=%2.2d, wrb_tail=%2.2d, wrb_count=%2.2d, time=%014.9f\n",
                      process_framecnt, write_ring_buffer.head_idx, write_ring_buffer.tail_idx, write_ring_buffer.count, 
                      realtime(&time_now) -start_realtime );
                    syslogPrint(data, SYSLOG_DATA);    
    
                }            
                // frame enqueued in the write buffer free the capture ring buffer positions
                rb_remove(&ring_buffer,FRAMES_PER_CAPTURE);
            } /* while ring_buffer.count */
        } else {
            clock_gettime(CLOCK_MONOTONIC, &time_now);
            sprintf(data, "S2: Write Ring buffer: Not enough frames in buffer to process: process_framecnt: %5.5d => WRITE RB head: %2.2d , tail %2.2d, count %2.2d, Time: %014.9f\n",
                  process_framecnt, write_ring_buffer.head_idx, write_ring_buffer.tail_idx, write_ring_buffer.count, 
                  realtime(&time_now) -start_realtime );
            syslogPrint(data, SYSLOG_DATA);
        } /*if ring_buffer count ... else */
    }/*read framecnt > 0*/
    return process_framecnt;
}

/// @brief convert the frames in the ring-buffer to original format to rgb and asve them
/// @param  none
/// @return saved frames counter
int seq_frame_store(void)
{
    int frames_num=write_ring_buffer.count;
    /*buffer for log*/
    char data[MAX_STRING_LEN];
    for(int frame_cnt= frames_num; frame_cnt>0; frame_cnt--){
        clock_gettime(CLOCK_MONOTONIC, &time_now);       
        // convert the image to rgb and save result in the scratch pad
        process_image((void *)&(write_ring_buffer.save_frame[write_ring_buffer.head_idx].frame[0]),
             FRAME_SIZE);
         
        save_image(scratchpad_buffer, FRAME_SIZE, &time_now);
        // frame processed  free the write ring buffer positions
        rb_remove(&write_ring_buffer,1);
        // save log message for debug
        sprintf(data, "S3: Store            :   store_framecnt=%5.5d, wrb_head_idx=%2.2d, wrb_tail=%2.2d, wrb_count=%2.2d, time=%014.9f\n",
                save_framecnt, write_ring_buffer.head_idx, write_ring_buffer.tail_idx, write_ring_buffer.count,
                realtime(&time_now) -start_realtime );
        syslogPrint(data, SYSLOG_DATA);    

    }

    return save_framecnt;
}


static void stop_capturing(void)
{
    enum v4l2_buf_type type;

    clock_gettime(CLOCK_MONOTONIC, &time_stop);
    fstop = (double)time_stop.tv_sec + (double)time_stop.tv_nsec / 1000000000.0;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if(-1 == xioctl(camera_device_fd, VIDIOC_STREAMOFF, &type))
		    errno_exit("VIDIOC_STREAMOFF");

    printf("capture stopped\n");
}


static void start_capturing(void)
{
        unsigned int i;
        enum v4l2_buf_type type;

	printf("will capture to %d buffers\n", n_buffers);

        for (i = 0; i < n_buffers; ++i) 
        {
                printf("allocated buffer %d\n", i);

                CLEAR(frame_buf);
                frame_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                frame_buf.memory = V4L2_MEMORY_MMAP;
                frame_buf.index = i;

                if (-1 == xioctl(camera_device_fd, VIDIOC_QBUF, &frame_buf))
                        errno_exit("VIDIOC_QBUF");
        }

        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (-1 == xioctl(camera_device_fd, VIDIOC_STREAMON, &type))
                errno_exit("VIDIOC_STREAMON");

}


static void uninit_device(void)
{
        unsigned int i;

        for (i = 0; i < n_buffers; ++i)
                if (-1 == munmap(buffers[i].start, buffers[i].length))
                        errno_exit("munmap");

        free(buffers);
}


static void init_mmap(char *dev_name)
{
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = DRIVER_MMAP_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

	printf("init_mmap req.count=%d\n",req.count);



    rb_init(&ring_buffer,       FRAME_SIZE, READ_RING_BUFFER_SIZE );
    rb_init(&write_ring_buffer, FRAME_SIZE, WRITE_RING_BUFFER_SIZE);
    rb_init(&edge_ring_buffer,  FRAME_SIZE, EDGE_RING_BUFFER_SIZE );



    if (-1 == xioctl(camera_device_fd, VIDIOC_REQBUFS, &req)) 
    {
            if (EINVAL == errno) 
            {
                    fprintf(stderr, "%s does not support "
                             "memory mapping\n", dev_name);
                    exit(EXIT_FAILURE);
            } else 
            {
                    errno_exit("VIDIOC_REQBUFS");
            }
    }

    if (req.count < 2) 
    {
            fprintf(stderr, "Insufficient buffer memory on %s\n", dev_name);
            exit(EXIT_FAILURE);
    }
	else
	{
	    printf("Device supports %d mmap buffers\n", req.count);

	    // allocate tracking buffers array for those that are mapped
            buffers = calloc(req.count, sizeof(*buffers));


	    // set up double buffer for frames to be safe with one time malloc her or just declare

	}

        if (!buffers) 
        {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < req.count; ++n_buffers) 
	{
                CLEAR(frame_buf);

                frame_buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                frame_buf.memory      = V4L2_MEMORY_MMAP;
                frame_buf.index       = n_buffers;

                if (-1 == xioctl(camera_device_fd, VIDIOC_QUERYBUF, &frame_buf))
                        errno_exit("VIDIOC_QUERYBUF");

                buffers[n_buffers].length = frame_buf.length;
                buffers[n_buffers].start =
                        mmap(NULL /* start anywhere */,
                              frame_buf.length,
                              PROT_READ | PROT_WRITE /* required */,
                              MAP_SHARED /* recommended */,
                              camera_device_fd, frame_buf.m.offset);

                if (MAP_FAILED == buffers[n_buffers].start)
                        errno_exit("mmap");

                printf("mappped buffer %d\n", n_buffers);
        }
}


static void init_device(char *dev_name)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_control cam_ctrl;
    unsigned int min;

    if (-1 == xioctl(camera_device_fd, VIDIOC_QUERYCAP, &cap))
    {
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\n",
                     dev_name);
            exit(EXIT_FAILURE);
        }
        else
        {
                errno_exit("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
    {
        fprintf(stderr, "%s is no video capture device\n",
                 dev_name);
        exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        fprintf(stderr, "%s does not support streaming i/o\n",
                 dev_name);
        exit(EXIT_FAILURE);
    }


    /* Select video input, video standard and tune here. */


    CLEAR(cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl(camera_device_fd, VIDIOC_CROPCAP, &cropcap))
    {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(camera_device_fd, VIDIOC_S_CROP, &crop))
        {
            switch (errno)
            {
                case EINVAL:
                    /* Cropping not supported. */
                    break;
                default:
                    /* Errors ignored. */
                        break;
            }
        }

    }
    else
    {
        /* Errors ignored. */
    }


    CLEAR(fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (force_format)
    {
        printf("FORCING FORMAT HxV: %dx%d\n",HRES,VRES);
        fmt.fmt.pix.width       = HRES;
        fmt.fmt.pix.height      = VRES;

        // Specify the Pixel Coding Formate here

        // This one works for Logitech C200
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;

        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_UYVY;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_VYUY;

        // Would be nice if camera supported
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;

        //fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;

        if ( xioctl(camera_device_fd, VIDIOC_S_FMT, &fmt) < 0)
                errno_exit("VIDIOC_S_FMT");

        /* Note VIDIOC_S_FMT may change width and height. */
    }
    else
    {
        printf("ASSUMING FORMAT\n");
        /* Preserve original settings as set by v4l2-ctl for example */
        if (-1 == xioctl(camera_device_fd, VIDIOC_G_FMT, &fmt))
                    errno_exit("VIDIOC_G_FMT");
    }
    /*check values set*/
    if( fmt.fmt.pix.width != HRES ||  fmt.fmt.pix.height != VRES){
        printf("Actual frame format different from expected\nExpected: %dx%d\n"
        "Actual: %dx%d\n",VRES,HRES,fmt.fmt.pix.height,fmt.fmt.pix.width);
        exit(-1);
    }
    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
            fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
            fmt.fmt.pix.sizeimage = min;

    init_mmap(dev_name);
    
    //disable autofocus because it interferes with 
    //differentiation of images
    cam_ctrl.id=V4L2_CID_FOCUS_AUTO;
    cam_ctrl.value = 0;
    if (-1 == ioctl(camera_device_fd,  VIDIOC_S_CTRL, &cam_ctrl))
    {  
        perror("Error disabling autofocus: \n");
        //close(camera_device_fd);
    }
    // Set the exposure mode to manual
    cam_ctrl.id = V4L2_CID_EXPOSURE_AUTO;
    cam_ctrl.value = V4L2_EXPOSURE_MANUAL;
    if (ioctl(camera_device_fd, VIDIOC_S_CTRL, &cam_ctrl) == -1) {
        perror("Error setting exposure mode");
        //close(camera_device_fd);
    }
    // Set the exposure value (adjust as needed)
    cam_ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
    cam_ctrl.value = 100; // Example: set exposure to 100 (adjust as needed)
    if (ioctl(camera_device_fd, VIDIOC_S_CTRL, &cam_ctrl) == -1) {
        perror("Error setting exposure value");
    }

}


static void close_device(void)
{
        if (-1 == close(camera_device_fd))
                errno_exit("close");

        camera_device_fd = -1;
}


static void open_device(char *dev_name)
{
        struct stat st;

        if (-1 == stat(dev_name, &st)) {
                fprintf(stderr, "Cannot identify '%s'=%d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }

        if (!S_ISCHR(st.st_mode)) {
                fprintf(stderr, "%s is no device\n", dev_name);
                exit(EXIT_FAILURE);
        }

        camera_device_fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

        if (-1 == camera_device_fd) {
                fprintf(stderr, "Cannot open '%s'=%d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
}



int v4l2_frame_acquisition_initialization(char *dev_name)
{
    // initialization of V4L2
    open_device(dev_name);
    init_device(dev_name);

    start_capturing();
}


int v4l2_frame_acquisition_shutdown(void)
{
    // shutdown of frame acquisition service
    stop_capturing();

    printf("Total capture time=%lf, for %d frames, %lf FPS\n",
       (fstop-fstart), read_framecnt+1, ((double)read_framecnt / (fstop-fstart)));

    uninit_device();
    close_device();
    fprintf(stderr, "\n");
    return 0;
}

