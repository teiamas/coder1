**Brief**: this project contains the code for an application that acquires the images of a stopwatch synchronizes to it, saves an images of each stopwatch tick every thenth of a second and applyies a NPP Laplacian filter  to it and a CNN to recongnize the time and save it.

**Credits**: this project has been derived from the example Nvidia project boxFilterNPP provided in the Coursera "CUDA at scale for enterprise"  and from material supplied in the Coursera course of Real Time Systems by Sam Siewert.

**Minimum spec**: SM 2.0, a UVC camera with at least 30 frames/sec, a Jetson Nano for the cuBLas and cuDNN, and a PC for the CNN Training.

**Input**: the images acquired by a Camera filming a stopwatch.
**Output**: ppm images of the stopwatch synchornized at each tick of a tenth of second (ppm format), their elaboration with a Lapacian filter saved in pgm format and a csv file with the image name and the time displayed in the image.

**Directory structure**:
project root
    bin/aarch64/linux/   : it contains the binaries for the Jetson Nano
        debug
            seqv4l2      : the executable compiled in debug format
            time_net.trt : the parameters for the CNN
            frames       : folder containing the acuired images
        release
            seqv4l2      : the executable compiled in release format
            time_net.trt : the parameters for the CNN            
            frames       : folder containing the acuired images
    project 
        Common  : NPP common code examples provided by Nvidia
        seqval2 : the directory containing the source code for the program
            .vscode : contains the json files I used for the development with vscode, they are just examples and requires the creation in the user directory of a sh script containing the launch of the cuda-dbg using sudo privileges.
    train_CNN : contains the code for the PC to train the CNN with the images acquired by the camera attached to Jetson Nano
        dataset:
            frames: contains the images to be used to train the CNN
            automatic_label.py: utility script used to create a csv file from where the train script reads the names of zip files
                                containing the pgm images
            overall.py        : utility scripts to merge the zip files contianing the images in one
            zips              : the directory containing the zips with the images, the zips names are in the format mm_ss_d.zip in order to 
                                be correctly processed by the atuomatic_label.py script
        
**build and run**: from the seqval2 directory execute the command:
    "make clean build run"
    it will clean temporary files and any images that if already present build the code copy it bin target dir and runs it with sudo privileges (needed to change the scheduler policy)

**Description**:
    The program aquires the images at 30Hz rate. Every 10Hz selects the image that among the last three has the larger difference between it and the previous one because it corresponds to the images when the tick of 1/10 of second has been displayed. It pushes the images in ring buffer for another task to be saved in ppm format and to another to be converted in gray format and filtered by the GPU Kernel by a NPP Laplacian filter and saved in pgm format.
    **seqv4l2.c**: contains the main routine that changes the scheduler policy to FIFO and creates the threads and launches them
    **capturelib.c**: contains the routines to initialize the camera and capture the images and save them
    **laplacian_filter.cu"**: contains the Kernel that converts the yuyv images in grey format and applies the Laplace filter.
    **my_utility.c**: contains helper routines to write in the log file and for time conversion.
    **NPP_utils.c"**: contains a roiutine to display info about NPP library and check the Cuda device
    **ring_buffer.c**: contains the routines used to implement the ring buffers.






