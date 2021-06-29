/*
 * @Email: yueshangChang@gmail.com
 * @Author: nanmi
 * @Date: 2021-06-29 15:58:00
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-05-22 11:49:13
 */

#include "Trt.h"
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"

class InputParser{                                                              
    public:                                                                     
        InputParser (int &argc, char **argv){                                   
            for (int i=1; i < argc; ++i)                                        
                this->tokens.push_back(std::string(argv[i]));                   
        }                                                                       
        const std::string& getCmdOption(const std::string &option) const{       
            std::vector<std::string>::const_iterator itr;                       
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option); 
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){      
                return *itr;                                                    
            }                                                                   
            static const std::string empty_string("");                          
            return empty_string;                                                
        }                                                                       
        bool cmdOptionExists(const std::string &option) const{                  
            return std::find(this->tokens.begin(), this->tokens.end(), option)  
                   != this->tokens.end();                                       
        }                                                                       
    private:                                                                    
        std::vector <std::string> tokens;                                       
};  

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t--onnx\t\tinput onnx model, must specify\n"
              << "\t--batch_size\t\tdefault is 1\n"
              << "\t--mode\t\t0 for fp32 1 for fp16 2 for int8, default is 0\n"
              << "\t--engine\t\tsaved path for engine file, if path exists, "
                  "will load the engine file, otherwise will create the engine file "
                  "after build engine. dafault is empty\n"
              << "\t--calibrate_data\t\tdata path for calibrate data which contain "
                 "npz files, default is empty\n"
              << "\t--gpu\t\tchoose your device, default is 0\n"
              << "\t--dla\t\tset dla core if you want with 0,1..., default is -1(not enable)\n"
              << std::endl;
}

// image resize to target shape
struct Location
{
    int w;
    int h;
    int x;
    int y;
    cv::cuda::GpuMat Img;
};

Location ImageProcess(cv::Mat &srcImage, const int &th, const int &tw, const int &type)
{
    int w, h, x, y;
    float r_w = (float)(tw / (srcImage.cols*1.0));
    float r_h = (float)(th / (srcImage.rows*1.0));
    if (r_h > r_w) {
        w = tw;
        h = (int)(r_w * srcImage.rows);
        x = 0;
        y = (th - h) / 2;
    } else {
        w = (int)(r_h * srcImage.cols);
        h = th;
        x = (tw - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, type);
    cv::cuda::GpuMat input(srcImage);
    cv::cuda::GpuMat output(re);
    input.upload(srcImage);
    cv::cuda::resize(input, output, re.size());

    Location out{w, h, x, y, output};
    return out;
}

int main(int argc, char** argv) {
    // parse args
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    InputParser cmdparams(argc, argv);

    const std::string& onnx_path = cmdparams.getCmdOption("--onnx");
    
    std::vector<std::string> custom_outputs;
    const std::string& custom_outputs_string = cmdparams.getCmdOption("--custom_outputs");
    std::istringstream stream(custom_outputs_string);
    if(custom_outputs_string != "") {
        std::string s;
        while (std::getline(stream, s, ',')) {
            custom_outputs.push_back(s);
        }
    }

    int run_mode = 0;
    const std::string& run_mode_string = cmdparams.getCmdOption("--mode");
    if(run_mode_string != "") {
        run_mode = std::stoi(run_mode_string);
    }

    const std::string& engine_file = cmdparams.getCmdOption("--engine");

    int batch_size = 1;
    const std::string& batch_size_string = cmdparams.getCmdOption("--batch_size");
    if(batch_size_string != "") {
        batch_size = std::stoi(batch_size_string);
    }

    const std::string& calibrateDataDir = cmdparams.getCmdOption("--calibrate_data");
    const std::string& calibrateCache = cmdparams.getCmdOption("--calibrate_cache");

    int device = 0;
    const std::string& device_string = cmdparams.getCmdOption("--gpu");
    if(device_string != "") {
        device = std::stoi(device_string);
    }

    int dla_core = -1;
    const std::string& dla_core_string = cmdparams.getCmdOption("--dla");
    if(dla_core_string != "") {
        dla_core = std::stoi(dla_core_string);
    }

    // build engine
    Trt* onnx_net = new Trt();
    onnx_net->SetDevice(device);
    onnx_net->SetDLACore(dla_core);
    if(calibrateDataDir != "" || calibrateCache != "") {
        onnx_net->SetInt8Calibrator("Int8EntropyCalibrator2", batch_size, calibrateDataDir, calibrateCache);
    }
    onnx_net->CreateEngine(onnx_path, engine_file, custom_outputs, batch_size, run_mode);
    
    // input 
    const int BATCH_SIZE = 1;
    const int INPUT_H = 789;
    const int INPUT_W = 1039;
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    cv::namedWindow("low light video enhanced");
    bool test_img = true;
    bool test_video = false;

    if (test_img)
    {
        const char* file_names = "../image/test.jpg";
        cv::Mat img = cv::imread(file_names);
        if (img.empty()) std::cerr << "Read image failed!" << std::endl;

        auto time_start = std::chrono::steady_clock::now();
        int gpu = 1;
        int cpu = 0;
        if (gpu)
        {
            cv::Mat re;
            cv::cuda::GpuMat input;

            cv::cuda::GpuMat output_img;
            input.upload(img);

            cv::cuda::resize(input, output_img, cv::Size(INPUT_W, INPUT_H));
            
            cv::cuda::GpuMat flt_image(INPUT_H, INPUT_W, CV_32FC3);
            output_img.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
            
            std::vector<cv::cuda::GpuMat> chw;

            float* gpu_input = (float*)(onnx_net->mBinding[0]);

            for (size_t i = 0; i < 3; ++i)
            {
                chw.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_input + i * INPUT_W * INPUT_H));
            }
            cv::cuda::split(flt_image, chw);

            onnx_net->CopyFromHostToDevice(gpu_input, 0);
            
            // do inference
            auto time1 = std::chrono::steady_clock::now();
            onnx_net->Forward();
            auto time2 = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
            std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;

            // trt output
            float* gpu_output = (float*)(onnx_net->mBinding[1]);
            
            cv::cuda::GpuMat flt_image_out;
            cv::cuda::GpuMat out_put;

            std::vector<cv::cuda::GpuMat> chw_1;
            for (size_t i = 0; i < 3; ++i)
            {
                chw_1.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_output + i * INPUT_W * INPUT_H));
            }
            cv::cuda::merge(chw_1, out_put);

            cv::cuda::GpuMat image_out;
            out_put.convertTo(image_out, CV_32FC3, 1.f * 255.f);
            cv::cuda::resize(image_out, flt_image_out, img.size());
            
            cv::Mat dst;
            flt_image_out.download(dst);

            auto time_end = std::chrono::steady_clock::now();
            auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
            std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;

            cv::imwrite("../01_test_demo.jpg", dst);
            // cv::imshow("low light video enhanced", image_out.Img);
        }
        if (cpu)
        {
            cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
            cv::resize(img, out, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
            cv::Mat pr_img = out; // letterbox BGR to RGB

            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[i] = (float)(uc_pixel[2] / 255.0);
                    data[i + INPUT_H * INPUT_W] = (float)(uc_pixel[1] / 255.0);
                    data[i + 2 * INPUT_H * INPUT_W] = (float)(uc_pixel[0] / 255.0);
                    uc_pixel += 3;
                    ++i;
                }
            }

            std::vector<float> input(data, data + sizeof(data)/sizeof(float));
            onnx_net->CopyFromHostToDevice(input, 0);
                    // do inference
            auto time1 = std::chrono::steady_clock::now();
            onnx_net->Forward();
            auto time2 = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
            std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
            
            // get output
            std::vector<float> output;
            onnx_net->CopyFromDeviceToHost(output, 1);

            // post process
            int kElem = INPUT_H * INPUT_W;
            std::vector<float> rr(kElem);
            std::vector<float> gg(kElem);
            std::vector<float> bb(kElem);
            for (int j = 0; j < kElem; j++)
            {
                bb[j] = (float)(output[j] * 255.0);
                gg[j] = (float)(output[j + kElem] * 255.0);
                rr[j] = (float)(output[j + 2 * kElem] * 255.0);
            }

            cv::Mat channel[3];
            channel[0] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, rr.data());
            channel[1] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, gg.data());
            channel[2] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, bb.data());
            auto image = cv::Mat(INPUT_H, INPUT_W, CV_32FC3);
            cv::merge(channel, 3, image);
            cv::Mat output_image;
            cv::resize(image, output_image, img.size(), 0, 0, cv::INTER_LINEAR);

            auto time_end = std::chrono::steady_clock::now();
            auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
            std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;

            cv::imwrite("../01_test.jpg", output_image);
            // cv::imshow("low light video enhanced", image_out.Img);
        }
    }
    else if (test_video)
    {
        const char* video_url = "../file/test3.avi";
        cv::VideoCapture cap(video_url);
        if (!cap.isOpened()) std::cout << "video open failed!" << std::endl;
        cv::Mat img;
        while (1)
        {
            cap >> img;
            if (img.empty()) std::cerr << "Read image failed!" << std::endl;

            auto time_start = std::chrono::steady_clock::now();

            int gpu = 1;
            int cpu = 0;
            if (gpu)
            {
                std::cout << "==========================================" << std::endl;
                cv::cuda::GpuMat input;
                cv::cuda::GpuMat output_img;
                input.upload(img);

                cv::cuda::resize(input, output_img, cv::Size(INPUT_W, INPUT_H));
            
                cv::cuda::GpuMat flt_image(INPUT_H, INPUT_W, CV_32FC3);
                output_img.convertTo(flt_image, CV_32FC3, 1.f / 255.f);

                std::vector<cv::cuda::GpuMat> chw;
                float* gpu_input = (float*)(onnx_net->mBinding[0]);
                for (size_t i = 0; i < 3; ++i)
                {
                    chw.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_input + i * INPUT_W * INPUT_H));
                }
                cv::cuda::split(flt_image, chw);
                onnx_net->CopyFromHostToDevice(gpu_input, 0);

                // do inference
                auto time1 = std::chrono::steady_clock::now();
                onnx_net->Forward();
                auto time2 = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
                std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
                
                // get output
                float* gpu_output = (float*)(onnx_net->mBinding[1]);
                
                cv::cuda::GpuMat flt_image_out;
                cv::cuda::GpuMat out_put;

                std::vector<cv::cuda::GpuMat> chw_1;
                for (size_t i = 0; i < 3; ++i)
                {
                    chw_1.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_output + i * INPUT_W * INPUT_H));
                }
                cv::cuda::merge(chw_1, out_put);

                cv::cuda::GpuMat image_out;
                out_put.convertTo(image_out, CV_32FC3, 1.f * 255.f);
                cv::cuda::resize(image_out, flt_image_out, img.size());
                
                cv::Mat dst;
                flt_image_out.download(dst);

                auto time_end = std::chrono::steady_clock::now();
                auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
                std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;


                cv::imwrite("../files/01_1.jpg", dst);
                cv::imshow("low light video enhanced", dst);
                cv::waitKey(1);
            }
            if (cpu)
            {
                int w, h, x, y;
                float r_w = (float)(INPUT_W / (img.cols*1.0));
                float r_h = (float)(INPUT_H / (img.rows*1.0));
                if (r_h > r_w) {
                    w = INPUT_W;
                    h = (int)(r_w * img.rows);
                    x = 0;
                    y = (INPUT_H - h) / 2;
                } else {
                    w = (int)(r_h * img.cols);
                    h = INPUT_H;
                    x = (INPUT_W - w) / 2;
                    y = 0;
                }
                cv::Mat re(h, w, CV_8UC3);
                cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
                re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
                cv::resize(img, re, re.size());
                cv::Mat pr_img = out; // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[i] = (float)(uc_pixel[2] / 255.0);
                        data[i + INPUT_H * INPUT_W] = (float)(uc_pixel[1] / 255.0);
                        data[i + 2 * INPUT_H * INPUT_W] = (float)(uc_pixel[0] / 255.0);
                        uc_pixel += 3;
                        ++i;
                    }
                }

                std::vector<float> input(data, data + sizeof(data)/sizeof(float));
                onnx_net->CopyFromHostToDevice(input, 0);

                // do inference
                auto time1 = std::chrono::steady_clock::now();
                onnx_net->Forward();
                auto time2 = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
                std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
                
                // get output
                std::vector<float> output;
                onnx_net->CopyFromDeviceToHost(output, 1);
                
                // post process
                int kElem = INPUT_H * INPUT_W;
                std::vector<float> rr(kElem);
                std::vector<float> gg(kElem);
                std::vector<float> bb(kElem);
                for (int j = 0; j < kElem; j++)
                {
                    bb[j] = (float)(output[j] * 255.0);
                    gg[j] = (float)(output[j + kElem] * 255.0);
                    rr[j] = (float)(output[j + 2 * kElem] * 255.0);
                }

                cv::Mat channel[3];
                channel[2] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, rr.data());
                channel[1] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, gg.data());
                channel[0] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, bb.data());
                auto image = cv::Mat(INPUT_H, INPUT_W, CV_32FC3);
                cv::merge(channel, 3, image);

                cv::Rect rect(x, y, w, h);
                cv::Mat image_roi = image(rect);

                auto time_end = std::chrono::steady_clock::now();
                auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
                std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;

                
                cv::imwrite("../01_test.jpg", image);
                // cv::imshow("low light video enhanced", image_out.Img);
            }

        }
    
    }
    else
    {
        std::cout << "choose a mode test." << std::endl;
    }
    
    return 0;
}