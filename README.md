**Brief**: this project contains the code for an application that acquires the images of a stopwatch synchronizes to it, saves an images of each stopwatch tick every thenth of a second and applyies a NPP Laplacian filter  to it and a CNN to recongnize the time and save it in text format.

**Credits**: this project has been derived from the example Nvidia project boxFilterNPP provided in the Coursera "CUDA at scale for enterprise"  and from material supplied in the Coursera course of Real Time Systems by Sam Siewert.

**Minimum spec**: SM 2.0, a UVC camera with at least 30 frames/sec, a Jetson Nano for the cuBLas and cuDNN, and a PC for the CNN Training.

**Input**: the images acquired by a Camera filming a stopwatch.<br>

**Output**: ppm images of the stopwatch synchornized at each tick of a tenth of second (ppm format), their elaboration with a Lapacian filter saved in pgm format and a csv file with the image name and the time displayed in the image.

**Directory structure**:<br><br>
**project root**<br>
    - **bin/aarch64/linux/**  : it contains the binaries for the Jetson Nano<br>
        - **debug**<br>
            - **seqv4l2**      : the executable compiled in debug format<br>
            - **time_net.tr**t : the parameters for the CNN: not included in this repo<br>
            - **frames**       : folder containing the acquired images<br>
        - **release**<br>
            - **seqv4l2**      : the executable compiled in release format<br>
            - **time_net.trt** : the parameters for the CNN: not included in this repo<br>
            - **frames**      : folder containing the acquired images<br>
    - **project**<br>
        - **Common**  : NPP common code examples provided by Nvidia<br>
        - **seqval2** : the directory containing the source code for the program<br>
            - **.vscode** : contains the json files I used for the development with vscode, they are just examples and requires the creation in the user directory of a sh script containing the launch of the cuda-dbg using sudo privileges.<br>
            - **seqv4l2.c**: contains the main routine that changes the scheduler policy to FIFO and creates the threads and launches them<br>
            - **capturelib.c**: contains the routines to initialize the camera and capture the images and save them<br>
            - **laplacian_filter.cu**: contains the Kernel that converts the yuyv images in grey format and applies the Laplace filter.<br>
            - **my_utility.c**: contains helper routines to write in the log file and for time conversion.<br>
            - **NPP_utils.c**: contains a roiutine to display info about NPP library and check the Cuda device<br>
            - **ring_buffer.c**: contains the routines used to implement the ring buffers.<br>
            - **image2time_str.cpp**: runs the CNN to recognize the time from the images and save in text format (csv)<br>
            - **resizeAndNormalize.cu**: resize in this case is not needed they are all the same size, it converts from uInt8 to float  the image<br>
            - **gpu_frame_diff.cu**: computes the difference byte by byte of two immages.<br>  
    -**train_CNN** : contains the code for the PC to train the CNN with the images acquired by the camera attached to Jetson Nano<br>
        - **dataset**:<br>
            - **frames**: contains the images to be used to train the CNN<br>
                - **overall.py**        : utility scripts to merge the zip files contianing the images in one<br>
                - **zips**              : the directory containing the zips with the images, the zips names are in the format <br>
                                      mm_ss_d.zip in order to be correctly processed by the atuomatic_label.py script<br>
                - **automatic_label.py**: to create a csv file with the association of the file name and time shown in the image.<br>
                                      to work correctly the zip file must have the mm_ss_d.zip from the time displayed in the first image<br>
        -**display_pth.py**            : displays the main pth parameters<br>
        -**inspect_onnx_model**        : displays the main onnx parameters<br>
        -**my_CNN**                    : CNN struture definition<br>
        -**my_dataset.py**             : dataset struture definition<br>
        -**pth2onnx.py**               : convert from pth to onnx format because is more suitable for Jetson Nano<br>
        -**requirements.txt**          : list of python packages needed, note the torch pacage were installed the ones with Nvidia <br>
                                      Xuda support  to speed up. They have to be installed by end speciying the url<br>
        -**runinferences.py**          : to run inferences with the precomputed onnx parameters to compare PC result with Jetson <br>
                                      Nano<br>
        -**Train_CNN.py**              : to trin the CNN on the given dataset<br>
        -**readme.md**                 :this file.<br>
**build and run**: from the seqval2 directory execute the command:<br>
    To run correctly the program needs the trt file for the CNN parameters, but I couldn't include it because in the free version of GitHub are not allowed file bigger than 100 MB, and the parameters were 140 MB. Even zipping it doen't work. So if you want to run you have to train the CNN using the Python scripts included<br>
    /usr/src/tensorrt/bin/trtexec --onnx=time_net.onnx --saveEngine=time_net.trt 
    "make clean build run"
    it will clean temporary files and any images that if already present build the code copy it bin target dir and runs it with sudo privileges (needed to change the scheduler policy)


**Description**:
    The program aquires the images at 30Hz rate. Every 10Hz selects the image that among the last three has the larger difference between it and the previous one because it corresponds to the images when the tick of 1/10 of second has been displayed. It pushes the images in ring buffer for another task to be saved in ppm format and to another to be converted in gray format and filtered by the GPU Kernel by a NPP Laplacian filter and saved in pgm format. This last images is also passed to a CNN to recognize the time displayed and save it in a csv file.



