
#include "ring_buffer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @brief allocate a ring buffer memory together with its contained fames init the params
/// @param rb_ptr pointer to ring buffer status data
/// @param frame_size size of the frame lelement of each ring place
/// @param rb_size length of the ring
void rb_init(struct ring_buffer_t* rb_ptr, int frame_size, int rb_size){
    /* allocate ring*/
    rb_ptr->save_frame = (struct save_frame_t*)malloc(sizeof(struct save_frame_t)*rb_size);
    if (rb_ptr->save_frame == NULL) {
        printf("Ring buffer not allocated: %ld\n",sizeof(struct save_frame_t)*rb_size);
        exit(0);
    }
    else {
 
        // Memory has been successfully allocated
        printf("Ring buffer allocated.: %ld\n",sizeof(struct save_frame_t)*rb_size);
    }   
    /*init buffer cursors*/
    rb_ptr->tail_idx  = 0;
	rb_ptr->head_idx  = 0;
	rb_ptr->count     = 0;
	rb_ptr->ring_size = rb_size;
    /*for each element in the ring allocate a buffer*/
    for( int idx=0; idx < rb_ptr->ring_size; idx++){
        unsigned char* frame  = (unsigned char*)malloc(frame_size);
        if (frame == NULL) {
            printf("Frame %d not allocated: %d\n",idx, frame_size);
            exit(0);
        }
        rb_ptr->save_frame[idx].frame=frame;
    }
 
}
/// @brief push a frame in the ring buffer
/// @param rb_ptr pointer to the ring buffer
/// @param data_ptr pointer to the frame data to be inserted
/// @param bytesused number of bytes in the input buffer
/// @param framecnt input frame counter copied for traceability
/// @param time_stamp time when the frame was acquired
void rb_insert(struct ring_buffer_t* rb_ptr, void* data_ptr, int bytesused,int framecnt,
             struct timespec* time_stamp){
        memcpy((void *)&(rb_ptr->save_frame[rb_ptr->tail_idx].frame[0]), 
                        data_ptr, 
                        bytesused);
        sprintf(rb_ptr->save_frame[rb_ptr->tail_idx].identifier_str,"%d",framecnt);
        rb_ptr->save_frame[rb_ptr->tail_idx].time_stamp = *time_stamp;
        rb_ptr->tail_idx = (rb_ptr->tail_idx + 1) % rb_ptr->ring_size;
        rb_ptr->count++;

}
/// @brief free first in element place in the ring buffer
/// @param rb_ptr : pointer to ring baffer data struture
/// @param num : number of elements to free
void rb_remove(struct ring_buffer_t* rb_ptr,int num) {
    rb_ptr->head_idx = (rb_ptr->head_idx + num) % rb_ptr->ring_size;
    (rb_ptr->count)-= num;
}

