#ifndef SEQV4L2_CAPTURE_LIB_H_
#define SEQV4L2_CAPTURE_LIB_H_

//resolution in pixels of the image used
#define HRES (640)
#define VRES (480)

/// @brief dequeue the read frame from kernel buffers
/// @brief insert in the ring buffer and start next frame read
/// @param  none
/// @return read frames counter

extern int seq_frame_read(void);

/// @brief search the frame with the maximum difference between itslef and the one before it
/// @brief when found enqueue it
/// @param  none
/// @return read frames counter
extern int enque_max_diff_frame(void );

/// @brief convert the frames in the ring-buffer to original format to rgb and asve them
/// @param  none
/// @return saved frames counter
extern int seq_frame_store(void);

/// @brief takes data from edge ring buffer, transforms the image applying the Laplacian filter
/// @brief and so highlighting the edges and  save it
/// @param none
/// @return the number of images processed
extern int frame_edges_detection(void);

#endif