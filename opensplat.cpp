// Dependencies
#include <filesystem> // For file system operations
#include <nlohmann/json.hpp> // For JSON parsing and serialization
#include "opensplat.hpp" // Header for Gaussian Splats model
#include "input_data.hpp" // Header for input data handling
#include "utils.hpp" // Utility functions
#include "cv_utils.hpp" // OpenCV utilities
#include "constants.hpp" // Global constants
#include <cxxopts.hpp> // Command-line options parser
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mach/mach.h>
// #include <Metal/Metal.h>

// Namespace
namespace fs = std::filesystem; // Aliasing std::filesystem namespace for easier usage
using namespace torch::indexing; // Importing torch tensor indexing for easier tensor manipulation

double getMemoryUsage() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error retrieving memory info" << std::endl;
        return 0;
    }
    return info.resident_size / (1024.0 * 1024.0 * 1024.0 * 8.0) * 100.0;
}

int main(int argc, char *argv[]){
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();

    // Create a cxxopts::Options class, used to define and parse command line arguments
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    // Define command line arguments with descriptions and default values
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("keep-crs", "Retain the project input's coordinate reference system")
        ("cpu", "Force CPU execution")
        
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("2"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value("100"))
        ("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value("500"))
        ("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value("30"))
        ("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value("0.0002"))
        ("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value("0.01"))
        ("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value("4000"))
        ("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value("0.05"))

        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    // Positional argument handling and help message setup
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odm project path]");
    // Command line arguments parsing and error handling
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        // Print error and usage on invalid argument input
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    // Version or help requested, print and exit
    if (result.count("version")){
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    // Parameters from user input
    const std::string projectRoot = result["input"].as<std::string>(); // Input path
    const std::string outputScene = result["output"].as<std::string>(); // Output path
    const int saveEvery = result["save-every"].as<int>(); // Save scene frequency
    const bool validate = result.count("val") > 0 || result.count("val-render") > 0; // Validation flag
    const std::string valImage = result["val-image"].as<std::string>(); // Image withheld for validation
    const std::string valRender = result["val-render"].as<std::string>(); // Directory for validation rendering
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender); // Create validation dir if not exists
    const bool keepCrs = result.count("keep-crs") > 0; // Retain CRS flag
    const float downScaleFactor = (std::max)(result["downscale-factor"].as<float>(), 1.0f); // Input downscale factor
    const int numIters = result["num-iters"].as<int>(); // Number of iterations
    const int numDownscales = result["num-downscales"].as<int>(); // Number of image downscales
    const int resolutionSchedule = result["resolution-schedule"].as<int>(); // Resolution update frequency
    const int shDegree = result["sh-degree"].as<int>(); // Max spherical harmonics degree
    const int shDegreeInterval = result["sh-degree-interval"].as<int>(); // Spherical harmonics update interval
    const float ssimWeight = result["ssim-weight"].as<float>(); // SSIM weight for loss calculation
    const int refineEvery = result["refine-every"].as<int>(); // Refinement frequency
    const int warmupLength = result["warmup-length"].as<int>(); // Warmup length before refinement
    const int resetAlphaEvery = result["reset-alpha-every"].as<int>(); // Reset opacity every these many refinements
    const float densifyGradThresh = result["densify-grad-thresh"].as<float>(); // Gradient threshold for splitting
    const float densifySizeThresh = result["densify-size-thresh"].as<float>(); // Size threshold for splitting/duplication
    const int stopScreenSizeAt = result["stop-screen-size-at"].as<int>(); // Stop splitting after this many steps
    const float splitScreenSize = result["split-screen-size"].as<float>(); // Split screen size threshold

    // Device selection
    torch::Device device = torch::kCPU; // Default to CPU
    int displayStep = 10; // Display loss update frequency

    if (torch::hasCUDA() && result.count("cpu") == 0) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA; // Use CUDA if available
    } else if (torch::hasMPS() && result.count("cpu") == 0) {
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS; // Use MPS (Apple Silicon) if available
    }else{
        std::cout << "Using CPU" << std::endl;
        displayStep = 1; // Lower display frequency for CPU
    }

    try{
        // Load input data from the specified project path
        InputData inputData = inputDataFromX(projectRoot);

        // Parallel load and downscale images
        parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&downScaleFactor](Camera &cam){
            cam.loadImage(downScaleFactor); // Load and downscale image
        });

        // Withhold a validation camera if necessary
        auto t = inputData.getCameras(validate, valImage);
        std::vector<Camera> cams = std::get<0>(t);
        Camera *valCam = std::get<1>(t);

        // Initialize Gaussian Splatting model with user-defined parameters
        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
                    refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                    numIters, keepCrs,
                    device);

        // Create a list of camera indices for training
        std::vector< size_t > camIndices( cams.size() );
        std::iota( camIndices.begin(), camIndices.end(), 0 );
        InfiniteRandomIterator<size_t> camsIter( camIndices );

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Preparation elapsed " << elapsed.count() << " mm" << std::endl;

        double memory_usage = 0;

        // id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        // if (!device) {
        //     std::cout << "This device does not support Metal." << std::endl;
        //     return -1;
        // }
        // id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        // id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        // double gpuStartTime, gpuEndTime;

        int imageSize = -1;
        // Main optimization loop
        for (size_t step = 1; step <= numIters; step++){
            if (step == 9){
                std::cout << std::setw(5) << "Step" 
                    << std::setw(10) << "Loss" 
                    << std::setw(10) << "Time (ms)" 
                    << std::setw(15) << "Memory (%)" 
                    << std::setw(10) << "Spl Time"
                    << std::setw(10) << "Dup Time"
                    << std::setw(10) << "Con Time"
                    << std::setw(10) << "Cull Time"
                    << std::setw(10) << "Add Gau"
                    << std::setw(10) << "Cull Gau"
                    << std::setw(10) << "Total Gau" << std::endl;
                std::cout << std::setw(5) << "----" 
                    << std::setw(10) << "---------" 
                    << std::setw(10) << "---------" 
                    << std::setw(15) << "--------------"
                    << std::setw(10) << "---------"
                    << std::setw(10) << "---------"
                    << std::setw(10) << "---------"
                    << std::setw(10) << "---------"
                    << std::setw(10) << "---------"
                    << std::setw(10) << "---------"
                    << std::setw(10) << "---------" << std::endl;
            }
            
            if (step % displayStep == 0){
                std::cout << std::setw(5) << step;
                start = std::chrono::high_resolution_clock::now();
                // gpuStartTime = commandBuffer.GPUStartTime;
            } 

            Camera& cam = cams[ camsIter.next() ];

            model.optimizersZeroGrad();

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
            gt = gt.to(device);

            torch::Tensor mainLoss = model.mainLoss(rgb, gt, ssimWeight);
            mainLoss.backward();
            
            if (step % displayStep == 0) std::cout << std::setw(10) << mainLoss.item<float>();

            model.optimizersStep();
            model.schedulersStep(step);
            auto [doDensification, split_elapsed, duplicate_elapsed, concatenation_elapsed, cull_elapsed, add_gaussian, cull_gaussian, remain_gaussian] = model.afterTrain(step);

            if (saveEvery > 0 && step % saveEvery == 0){
                fs::path p(outputScene);
                model.save((p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string()));
            }

            if (!valRender.empty() && step % 10 == 0){
                torch::Tensor rgb = model.forward(*valCam, step);
                cv::Mat image = tensorToImage(rgb.detach().cpu());
                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                cv::imwrite((fs::path(valRender) / (std::to_string(step) + ".png")).string(), image);
            }

            if (step % displayStep == 0){
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                std::cout << std::setw(10) << elapsed.count();

                // [commandBuffer commit];
                // [commandBuffer waitUntilCompleted];
                // gpuEndTime = commandBuffer.GPUEndTime;
                // double gpuDuration = (gpuEndTime - gpuStartTime) * 1000;
                // std::cout << std::setw(10) << gpuDuration;

                memory_usage = getMemoryUsage();
                if (doDensification){
                    std::cout << std::setw(15) << memory_usage 
                        << std::setw(10) << split_elapsed.count()
                        << std::setw(10) << duplicate_elapsed.count()
                        << std::setw(10) << concatenation_elapsed.count()
                        << std::setw(10) << cull_elapsed.count()
                        << std::setw(10) << add_gaussian 
                        << std::setw(10) << cull_gaussian 
                        << std::setw(10) << remain_gaussian << std::endl;
                } else{
                    std::cout << std::setw(15) << memory_usage << std::endl;
                }
            }
        }

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene);
        // model.saveDebugPly("debug.ply");

        // Validate
        if (valCam != nullptr){
            torch::Tensor rgb = model.forward(*valCam, numIters);
            torch::Tensor gt = valCam->getImage(model.getDownscaleFactor(numIters)).to(device);
            std::cout << valCam->filePath << " validation loss: " << model.mainLoss(rgb, gt, ssimWeight).item<float>() << std::endl; 
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_elapsed = total_end - total_start;
        std::cout << "Total elapsed " << total_elapsed.count() << " mm" << std::endl;
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}