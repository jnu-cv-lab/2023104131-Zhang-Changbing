#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
if (argc != 2) {
    std::cout << "用法: " << argv[0] << " /home/chbing/Homework2/picture.jpg" << std::endl;
    return -1;
}

cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cout << "无法读取图片: " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "======= 图像基本信息 ======" << std::endl;
    std::cout << "宽度: " << image.cols << " 像素" << std::endl;
    std::cout << "高度: " << image.rows << " 像素" << std::endl;
    std::cout << "通道数: " << image.channels() << std::endl;
    std::cout << "数据类型: ";

    switch (image.depth()) {
        case CV_8U: std::cout << "CV_8U (8位无符号整数)" << std::endl; break;
        case CV_8S: std::cout << "CV_8S (8位有符号整数)" << std::endl; break;
        case CV_16U: std::cout << "CV_16U (16位无符号整数)" << std::endl; break;
        case CV_16S: std::cout << "CV_16S (16位有符号整数)" << std::endl; break;
        case CV_32S: std::cout << "CV_32S (32位有符号整数)" << std::endl; break;
        case CV_32F: std::cout << "CV_32F (32位浮点数)" << std::endl; break;
        case CV_64F: std::cout << "CV_64F (64位浮点数)" << std::endl; break;
        default: std::cout << "未知类型" << std::endl;
    }

    cv::namedWindow("原始图像", cv::WINDOW_NORMAL);
    cv::imshow("原始图像", image);

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::namedWindow("灰度图像", cv::WINDOW_NORMAL);
    cv::imshow("灰度图像", grayImage);

    std::string grayFilename = "gray_" + std::string(argv[1]);
    cv::imwrite(grayFilename, grayImage);
    std::cout << "灰度图像已保存为: " << grayFilename << std::endl;

    cv::Point center(image.cols/2, image.rows/2);
    if (image.channels() == 3) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(center);
        std::cout << "中心点 (" << center.x << ", " << center.y << ") 的BGR值: "
                  << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << std::endl;
    } else if (image.channels() == 1) {
        uchar pixel = image.at<uchar>(center);
        std::cout << "中心点 (" << center.x << ", " << center.y << ") 的像素值: "
                  << (int)pixel << std::endl;
    }

    int cropSize = 100;
    cv::Rect roi(0, 0, std::min(cropSize, image.cols), std::min(cropSize, image.rows));
    cv::Mat cropped = image(roi).clone();
    std::string cropFilename = "cropped_" + std::string(argv[1]);
    cv::imwrite(cropFilename, cropped);
    std::cout << "裁剪区域已保存为: " << cropFilename << std::endl;
    std::cout << "裁剪区域尺寸: " << cropped.cols << "x" << cropped.rows << std::endl;

    cv::namedWindow("裁剪区域（左上角）", cv::WINDOW_NORMAL);
    cv::imshow("裁剪区域（左上角）", cropped);

    std::cout << "\n按任意键关闭图像窗口..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}