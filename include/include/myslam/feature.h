#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include "myslam/common_include.h"
#include <opencv2/features2d.hpp>
#include <memory>

namespace myslam
{
    struct Frame;
    struct MapPoint;

    /**
     * 2D 特征点
     * 在三角化之后会被关联一个地图点
     */

    struct Feature
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr;
        cv::KeyPoint position_;        // 2D提取位置
        bool is_outlier_ = false;      // 是否为异常点
        bool is_on_left_image_ = true; // 标识是否提在左图，false为右图

        /*weak_ptr是弱智能指针对象，它不控制所指向对象生存期的智能指针，
         * 它指向由一个shared_ptr管理的智能指针。
         * 将一个weak_ptr绑定到一个shared_ptr对象，不会改变shared_ptr的引用计数。
         * 一旦最后一个所指向对象的shared_ptr被销毁，所指向的对象就会被释放，即使此时有weak_ptr指向该对象，所指向的对象依然被释放。
         *weak_ptr是为了配合shared_ptr而引入的一种智能指针，
         * 因为它不具有普通指针的行为，没有重载operator*和->,
         * 它的最大作用在于协助shared_ptr工作，像旁观者那样观测资源的使用情况。
         * weak_ptr可以从一个shared_ptr或者另一个weak_ptr对象构造，获得资源的观测权。
         * 但weak_ptr没有共享资源，它的构造不会引起指针引用计数的增加
         * weak_ptr可以使用一个非常重要的成员函数lock()从被观测的shared_ptr获得一个可用的shared_ptr对象,就像复制出一个样本一样
         */

        std::weak_ptr<Frame> frame_; // 持有该feature的frame
        /*
        added by caohm
        特征点和 MapPoint 是一一对应的：每个特征点在三维空间中有一个唯一的 MapPoint 对象表示其三维位置。
        当一个特征点被检测到后，如果在多视角中找到对应关系，就可以通过三角化计算出其在三维空间中的位置，生成一个 MapPoint 对象。
        Frame 类中有很多特征点：每个 Frame 对象包含该帧图像中检测到的所有特征点。
        不同 Frame 中的特征点可能是同一个 MapPoint：在不同 Frame 中检测到的特征点可能对应同一个 MapPoint 对象，这表示这些特征点都观测到了同一个三维点。
        */
        std::weak_ptr<MapPoint> map_point_; // 关联地图点

    public:
        Feature() {}

        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp) {}
    };
}

#endif // MYSLAM_FEATURE_H