#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;


namespace myslam{

    Dataset::Dataset(const std::string& dataset_path)
            : dataset_path_(dataset_path) {}


    /*
    added by caohm 
    P0: 7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 0.000000000000e+00 0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    P1: 7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 -3.798145000000e+02 0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    P2: 7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 4.688783000000e+01 0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 1.178601000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 6.203223000000e-03
    P3: 7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 -3.334597000000e+02 0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 1.930130000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.318498000000e-03
    这是一个初始投影矩阵，根据投影矩阵可以提取内参矩阵、变换矩阵、平移向量
    这个数据集中默认旋转矩阵是单位矩阵，所以可以使用一下简单的方法提取K,t,更加通用的方法是利用QR分解得到内参矩阵，旋转矩阵和平移向量
    利用通用方法和简单的方法提取出来的值是一样的
    P=K[R|t]  ==>  P=K[I|t] ==> P=[K|Kt] 
    R|t ==> 指的是 3*3的旋转矩阵又增加了一维平移向量，最终变成了 3*4矩阵， 同样 [K|Kt]表示 3*3的内参矩阵又增加了一维 Kt，最终变成了3*4矩阵
    */

    /*
    added by caohm
    光心是垂直于xy平面的一个点，在xy平面上会有一个光心投射点。从光心处画一根垂直于x轴的线，再画一根垂直于y轴的线，这两根线分别在x轴和y轴有A点和B点。
    光心投射点到A点的距离是 fx，到B点的距离是 fy。而 f 是光心到光心投射点的距离。
    */
    bool Dataset::Init()
    {
        ifstream fin(dataset_path_ + "/calib.txt");
        if (!fin) {
            LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
            return false;
        }

        for (int i = 0; i < 4; ++i){   //一共有P0,P1,P2,P3四个相机，这里就是要把四个相机的参数全部读取到
            //前三个字符是P0：所以这里定义了一个长度为3的字符数组，读完这三个字符后就遇到了第一个空格，fin将会跳过这个空格，读取参数
            char camera_name[3];
            for (int k = 0; k < 3; ++k) {
                fin >> camera_name[k];
            }

            //将相机后面对应的12个参数读入到projection_data[12]中
            double projection_data[12];
            for (int k = 0; k < 12; ++k) {
                fin >> projection_data[k];
            }

            //将projection_data[12]的参数分为两部分提取出来
            Mat33 K;
            K << projection_data[0], projection_data[1], projection_data[2],
                    projection_data[4], projection_data[5], projection_data[6],
                    projection_data[8], projection_data[9], projection_data[10];
            Vec3 t;
            t << projection_data[3], projection_data[7], projection_data[11];

            /*
            added by caohm Kt = A, 要想求出t，则两边左乘一个K逆即可
            */
            t = K.inverse() * t;
            ///t = K.inverse() * t,参考https://blog.csdn.net/yangziluomu/article/details/78339575
            
            /*
            added by caohm
            首先，我们需要了解相机投影模型。一个常见的相机投影模型可以用以下矩阵形式表示：
                u         fx 0 cx         Xc
                v   =     0  fy cy   *    Yc
                1         0  0  0         Zc
            
            当图像被缩小0.5倍时，图像坐标系中的每一个像素坐标 (u,v) 也会相应地缩小0.5倍。
            内参矩阵中的焦距 fx 和 fy 表示在相机坐标系中单位长度投影到图像平面上时的像素数。
            当图像缩小0.5倍时，同样的单位长度在图像平面上只占原来的0.5倍像素数。因此，缩小后的焦距应为原来的一半
            光学中心 cx 和 cy 表示图像平面上的中心点位置。缩小图像时，光学中心的坐标也相应地缩小0.5倍
            可以看到当图像被缩小时，图像坐标和内参矩阵中的参数都相应地缩小0.5倍，以保持几何关系的一致性。
            */
            K = K * 0.5;//因为前面你把读到的图像全部resize成了原来的一半，所以需要在内参矩阵上乘以0.5,将投影获得的像素坐标也变为原来的一半
            /*
            added by caohm 
            SE3: 特殊欧氏群（Special Euclidean Group）中的一个元素，通常用于描述三维空间中的刚体变换。刚体变换包括旋转和平移。
            SO3: 表示一个 3x3 的旋转矩阵。旋转矩阵是一个正交矩阵，并且其行列式为1。旋转矩阵用于表示刚体在三维空间中的旋转。
            默认构造的 SO3 对象：通常情况下，SO3() 创建的是一个单位旋转矩阵，即没有旋转。
            */
            /*
            added by caohm
            立体相机系统通常由两个相机组成，这两个相机之间有一定的水平距离，这个距离称为基线（baseline）。
            立体相机系统利用两个相机捕获的图像来计算深度信息，通过三角测量法来确定物体在三维空间中的位置。
            在立体相机系统中，基线是一个非常重要的参数，因为它直接影响深度计算的精度。基线越长，深度计算的精度越高。
            基线（baseline）可以通过平移向量来计算，主要是因为立体相机系统中相机的排列方式，以及基线在三维重建中的作用。
            在立体相机系统中，通常有两台相机并排放置，并且它们的光轴是平行的，位于同一个水平面上。基线是指两台相机光心（光学中心）之间的水平距离。
            在立体相机系统中，平移向量 t 表示从一个相机坐标系到另一个相机坐标系的平移。由于相机的光轴是平行的，且位于同一个水平面上，平移向量的水平方向分量就代表了两个相机之间的水平距离，即基线。
            对于KITTI数据集，由于相机的光轴是平行的，且位于同一个水平面上，旋转矩阵 R0  和 R1通常是单位矩阵 I，因此投影矩阵可以简化为：
            P0=K[I∣0], P1=K[I∣t],t 是平移向量，表示两个相机光心之间的水平距离。
            t.norm() 计算的是平移向量的范数，即基线长度
            基线是两个相机光心之间的水平距离，通常在立体视觉计算中使用。基线的计算需要两个相机的平移向量 t。
            在初始化每个相机时，只需要将该相机的平移向量 t 存储起来。在后续计算深度时，才会用到两个相机的平移向量。
            */
            Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                              t.norm(), SE3(SO3(), t)));
            cameras_.push_back(new_camera);
            LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();

        }


        fin.close();

        current_image_index_ = 0;//虽然类内已经对这个成员变量赋值了，但是这里还要写一个置0语句，是因为每次初始化之后都得给index置0
        return true;
    }

    Frame::Ptr Dataset::NextFrame()
    {

        boost::format fmt("%s/image_%d/%06d.png");

        //boost::format的相关内容和用法参考：
        ///https://blog.csdn.net/weixin_33802505/article/details/90594738?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control


        cv::Mat image_left, image_right;
        // read images
        image_left =
                cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                           cv::IMREAD_GRAYSCALE);
        image_right =
                cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                           cv::IMREAD_GRAYSCALE);

        if (image_left.data == nullptr || image_right.data == nullptr) {
            LOG(WARNING) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }


        //利用resize()函数改变图像尺寸
        //resize(InputArray src, OutputArray dst, Size dsize,double fx=0, double fy=0, int interpolation=INTER_LINEAR )
        /*
         * InputArray src ：输入，原图像，即待改变大小的图像；
           OutputArray dst： 输出，改变后的图像。这个图像和原图像具有相同的内容，只是大小和原图像不一样而已；
           dsize：输出图像的大小。
           如果这个参数不为0，那么就代表将原图像缩放到这个Size(width，height)指定的大小；如果这个参数为0，那么原图像缩放之后的大小就要通过下面的公式来计算：
           dsize = Size(round(fxsrc.cols), round(fysrc.rows))

          其中，fx和fy就是下面要说的两个参数，是图像width方向和height方向的缩放比例。
          fx：width方向的缩放比例，如果它是0，那么它就会按照(double)dsize.width/src.cols来计算；
          fy：height方向的缩放比例，如果它是0，那么它就会按照(double)dsize.height/src.rows来计算；

          interpolation：这个是指定插值的方式，图像缩放之后，肯定像素要进行重新计算的，就靠这个参数来指定重新计算像素的方式，有以下几种：
          INTER_NEAREST - 最邻近插值
          INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
          INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
          INTER_CUBIC - 4x4像素邻域内的双立方插值
          INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
         */

        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
                   cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
                   cv::INTER_NEAREST);


        auto new_frame = Frame::CreateFrame();
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
        current_image_index_++;
        return new_frame;

    }

}