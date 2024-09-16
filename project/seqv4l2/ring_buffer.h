#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stddef.h>
#include <sys/time.h>

struct save_frame_t
{
    unsigned char   *frame;
    struct timespec time_stamp;
    char identifier_str[80];
};


struct ring_buffer_t
{
    unsigned int ring_size;

    int tail_idx;
    int head_idx;
    int count;

    struct save_frame_t* save_frame;
};

/// @brief allocate a ring buffer memory together with its contained fames init the params
/// @param rb_ptr pointer to ring buffer status data
/// @param frame_size size of the frame lelement of each ring place
/// @param rb_size length of the ring
extern void rb_init(struct ring_buffer_t* rb_ptr,  int frame_size,int rb_size);

/// @brief push a frame in the ring buffer
/// @param rb_ptr pointer to the ring buffer
/// @param data_ptr pointer to the frame data to be inserted
/// @param bytesused number of bytes in the input buffer
/// @param framecnt input frame counter copied for traceability
/// @param time_stamp time when the frame was acquired
extern void rb_insert(struct ring_buffer_t* rb_ptr, void* data_ptr, int bytesused,
                    int framecnt,struct timespec* time_stamp);

/// @brief free first in element place in the ring buffer
/// @param rb_ptr pointer to ring baffer data struture
/// @param num number of elements to free
extern void rb_remove(struct ring_buffer_t* rb_ptr,int num);
#endif