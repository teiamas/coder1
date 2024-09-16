#include <cuda_runtime.h>
#include <npp.h>
#include <stdio.h>

/*Local module data initialized by the engage functions and used across subsequent images elaboration */
static unsigned char  lapalce_engaged=0; //used to check if the lapalacian has been engaged
static char*          d_yuyvImage;       //pointer to device (GPU) allocated yuyv image
static unsigned char* d_greyImage;       //pointer to device (GPU) allocated gray image
static unsigned char* d_laplImage;       //pointer to device (GPU) allocated Laplacian filtered image
static int numRows;                      //number of rows of the image 
static int numCols;                      //number of columns of the image
static int numElements;                  //number of elements of the image
static int yuyvImagesize;                // yuyv image sizes in pixels
static int grayImagesize;                // gray image sizes in pixels
static NppiSize oSizeROI;                // region of interest in pixel
static dim3 blockSize(16, 16);           // GPU's block size 
static dim3 gridSize;                    // GPU's Grid size



/// @brief displays the cuda error message and exits
/// @param err  cuda error code
/// @param msg  specific description to be appended to standard error message
extern  "C" __host__ void my_checkCudaErrors(cudaError_t err,  const char* msg);
__host__ void my_checkCudaErrors(cudaError_t err,  const char* msg) {
    if (err != cudaSuccess) {
        printf("Error in: %s\n%s\n",msg, cudaGetErrorString(err));
        exit(-1);
    }
}

/// @brief translate the grid box coordinates to progressive index
__device__
int getGlobalIdx_3D_3D(void) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

/// @brief convert the yuyv image to a gray image
/// @param yuyvImage   yuyv image to be converted
/// @param greyImage   image converted
/// @param numElements number of elements of the image
__global__ void yuyv2gray(const char* const yuyvImage, unsigned char* const greyImage, 
                                  int numElements) {
    int idx = getGlobalIdx_3D_3D();
    if ( idx > numElements ) {
        return;
    }

    float y1 = yuyvImage[2*idx];
    float y2 = yuyvImage[2*idx+1];

    // Compute grayscale value (average of Y components)
    float channelSum = (y1 + y2) / 2.0f;
    greyImage[idx] = static_cast<unsigned char>(channelSum);
}

/// @brief prpares the constant data and buffers used by the GPU to calc the laplacian
/// @param pheight   height in pixels
/// @param pwidth    width in pixels
extern "C" void engage_laplacian(int pheight, int pwidth);
void engage_laplacian(int pheight, int pwidth){

    numCols     = pwidth;
    numRows     = pheight;
    numElements = numCols * numRows;

    yuyvImagesize = numElements * 2 * sizeof(unsigned char);
    grayImagesize = numElements * 1 * sizeof(unsigned char);
    // Set the region of interest (ROI) size
    oSizeROI.width  = numCols;
    oSizeROI.height = numRows; 
    gridSize.x  =  (numCols + blockSize.x - 1) / blockSize.x;
    gridSize.y  =  (numRows + blockSize.y - 1) / blockSize.y;
    gridSize.z  =  1;

    my_checkCudaErrors( cudaMalloc((void**)&d_yuyvImage, yuyvImagesize),"Allocating yuyv data in the GPU");
    my_checkCudaErrors( cudaMalloc((void**)&d_greyImage, grayImagesize),"Allocating grey data in the GPU");
    my_checkCudaErrors( cudaMalloc((void**)&d_laplImage, grayImagesize),"Allocating grey data in the GPU");

    lapalce_engaged = 1;
}

/// @brief frees the resources engaged for the laplacian
extern "C" void disengage_laplacian(void);
void disengage_laplacian(void){
    lapalce_engaged = 0;
    my_checkCudaErrors(cudaFree(d_yuyvImage), "freeing yuyv data memory");
    my_checkCudaErrors(cudaFree(d_greyImage), "freeing grey data memory");
    my_checkCudaErrors(cudaFree(d_laplImage), "freeing grey data memory");
}

/// @brief applies the laplacian over the original image highlighting the edges  
/// @param img_src_ptr pointer to original yuyv image data
/// @param img_dst_ptr pointer to tranformed gray image data
/// @param pheight   height in pixels
/// @param pwidth    width in pixels
extern "C" void calc_laplacian(
    const unsigned char * const h_yuyvImage,
          unsigned char * const h_laplImage);

void calc_laplacian(
    const unsigned char * const h_yuyvImage,
          unsigned char * const h_laplImage ){
    if( lapalce_engaged == 0){
        printf("Call of Laplacian without engaging it\n");
        exit(-1);
    }
    // Assuming you have an input image 'oDeviceSrc' and an output image 'oDeviceDst'
    my_checkCudaErrors(cudaMemcpy(d_yuyvImage, h_yuyvImage, yuyvImagesize, cudaMemcpyHostToDevice),"Copying yuyv data 2 device");
    yuyv2gray<<<gridSize, blockSize>>>(d_yuyvImage,d_greyImage, numElements);
    //my_checkCudaErrors(cudaGetLastError(),"launching the conv2gray kernel");
    my_checkCudaErrors(cudaDeviceSynchronize(),"one task has failed in the conv2gray kernel");;
    //my_checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, grayImagesize * sizeof(char), cudaMemcpyDeviceToHost),"copying grey data to host");

    NppStatus npps =
        nppiFilterLaplace_8u_C1R	( 
        d_greyImage                , /*const Npp8u * pSrc,     */
        numCols                    , /*Npp32s 	     nSrcStep, */
        d_laplImage                , /*Npp8u * 	     pDst,     */
        numCols                    , /*Npp32s 	     nDstStep, */
        oSizeROI                   , /*NppiSize 	 oSizeROI, */
        NPP_MASK_SIZE_5_X_5          /*NppiMaskSize	 eMaskSize */
        );
    if( npps != NPP_SUCCESS){
        printf("Error in NPP filtering\n");
        exit(-1);
    }   
    my_checkCudaErrors(cudaMemcpy(h_laplImage, d_laplImage, grayImagesize, cudaMemcpyDeviceToHost),"copying grey data to host");

}

/// @brief returns the DEVICE address of the image filtered with lapalcian
/// @param none
/// @details returns the DEVICE address of the image filtered with lapalcian if the Laplacian is engaged,
///          exits otherwise
/// @return DEVICE address of the filtered image
extern "C" {

    unsigned char* get_d_laplacian_image(void){
        if( lapalce_engaged == 0){
            printf("Call of Laplacian without engaging it\n");
            exit(-1);
        } 
        return d_laplImage;
    
    } 
}

