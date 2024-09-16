// image2time_str.cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include "resizeAndNormalize.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINFO) return;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

//IO buffer number 1 input + 3 outputs
#define CNN_IO_BUFFER (4) 
#define INVALID_CNN_IO_BUFFER (CNN_IO_BUFFER+1)

Logger gLogger;

using namespace nvinfer1;
using namespace nvonnxparser;

IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;

static Npp8u* d_lapalced_img = nullptr;
static void* CNN_buffers[CNN_IO_BUFFER];

static int inputWidth=0;
static int inputHeight=0;

int inputIndex       = INVALID_CNN_IO_BUFFER;
int minutesIndex     = INVALID_CNN_IO_BUFFER;
int secondsIndex     = INVALID_CNN_IO_BUFFER;
int decisecondsIndex = INVALID_CNN_IO_BUFFER;

float* h_minutes = new float[60];
float* h_seconds = new float[60];
float* h_deciseconds = new float[10];

//outputcsv file with the recognized time strings
std::ofstream csvFile;
extern "C" {
    void reset_CNN(void) {
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
        context = nullptr;
        engine = nullptr;
        runtime = nullptr;
    }

    void engage_CNN(int width, int height, Npp8u* d_lap_img, char* csv_fname) {
        inputWidth = width;
        inputHeight = height;
        runtime = createInferRuntime(gLogger);
        std::ifstream file("time_net.trt", std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file!" << std::endl;
            exit(-1);
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();
        engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
        context = engine->createExecutionContext();
    
        // Allocate device memory
        if(d_lap_img == nullptr){
            std::cerr << "Error null device pointer to Laplaced image!" << std::endl;
            exit(-1);     
        } else {
            d_lapalced_img = d_lap_img;
        }

        inputIndex       = engine->getBindingIndex("input");
        minutesIndex     = engine->getBindingIndex("minutes");
        secondsIndex     = engine->getBindingIndex("seconds");
        decisecondsIndex = engine->getBindingIndex("deciseconds");
        std::cout << "inputIndex       : "<< inputIndex      << std::endl;  
        std::cout << "minutesIndex     : "<< minutesIndex    << std::endl;  
        std::cout << "secondsIndex     : "<< secondsIndex    << std::endl;  
        std::cout << "decisecondsIndex : "<< decisecondsIndex<< std::endl;  
        if( inputIndex   < CNN_IO_BUFFER && minutesIndex     < CNN_IO_BUFFER  && 
            secondsIndex < CNN_IO_BUFFER && decisecondsIndex < CNN_IO_BUFFER ){
            CUDA_CHECK(cudaMalloc(&CNN_buffers[inputIndex], inputWidth * inputHeight * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&CNN_buffers[minutesIndex], 60*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&CNN_buffers[secondsIndex], 60*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&CNN_buffers[decisecondsIndex], 10*sizeof(float)));
        } else {
            std::cerr << "Error Invalid index for CNN_buffers index!" << std::endl;
            exit(-1);
        }
        // open out file with strings corresponding to times
        csvFile.open(csv_fname, std::ios::app);
        if (!csvFile) {
            std::cerr << "Error opening CSV file!" << std::endl;
            exit(-1);
        }

    }


    void disengage_CNN() {
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
        context = nullptr;
        engine = nullptr;
        runtime = nullptr;
    
        // Free device memory
        CUDA_CHECK(cudaFree(CNN_buffers[inputIndex      ]));
        CUDA_CHECK(cudaFree(CNN_buffers[minutesIndex    ]));
        CUDA_CHECK(cudaFree(CNN_buffers[secondsIndex    ]));
        CUDA_CHECK(cudaFree(CNN_buffers[decisecondsIndex]));
        csvFile.close();
    }

    void runInference() {
    

        normalize(/*src*/d_lapalced_img,/*dst*/ reinterpret_cast<float*>(CNN_buffers[inputIndex]),/*width*/ inputWidth,/* height */ inputHeight);
        if( !context->executeV2(CNN_buffers) ){
            std::cerr << "Failure on V2 execution" << std::endl;
            exit(-1);
        } 
        CUDA_CHECK(cudaMemcpy(h_minutes, CNN_buffers[minutesIndex], 60 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_seconds, CNN_buffers[secondsIndex], 60 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_deciseconds, CNN_buffers[decisecondsIndex], 10 * sizeof(float), cudaMemcpyDeviceToHost));
        // Find the digit with the highest probability
        int predicted_minutes = std::distance(h_minutes, std::max_element(h_minutes, h_minutes + 60));
        int predicted_seconds = std::distance(h_seconds, std::max_element(h_seconds, h_seconds + 60));
        int predicted_deciseconds = std::distance(h_deciseconds, std::max_element(h_deciseconds, h_deciseconds + 10));
        // Save results to CSV
        if (csvFile.is_open()) {
            csvFile << predicted_minutes << "," << predicted_seconds << "," << predicted_deciseconds << std::endl;
        } else {
            std::cerr << "Error writing to CSV file!" << std::endl;
            exit(-1);
        }   
    }
}


    
