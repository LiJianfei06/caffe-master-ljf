#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
// 池化层，Layer类的子类，图像降采样，有三种Pooling方法：Max、Avx、Stochastic
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  // 显示构造函数
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 参数初始化，通过类PoolingParameter获取成员变量值，包括：
  // global_pooling_、kernel_h_、kernel_w_、pad_h_、pad_w_、stride_h_、stride_w_
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 调整top blobs的shape，并有可能会reshape rand_idx_或max_idx_；
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 获得Pooling layer的类型: Pooling
  virtual inline const char* type() const { return "Pooling"; }
  // 获得Pooling layer所需的bottom blobs的个数: 1
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // 获得Pooling layer所需的bottom blobs的最少个数: 1
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  // 获得Pooling layer所需的bottom blobs的最多个数: Max为2，其它(Avg, Stochastic)为1
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

 protected:
  // CPU实现Pooling layer的前向传播，仅有Max和Ave两种方法实现
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // GPU实现Pooling layer的前向传播，Max、Ave、Stochastic三种方法实现
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CPU实现Pooling layer的反向传播，仅有Max和Ave两种方法实现
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // GPU实现Pooling layer的反向传播，Max、Ave、Stochastic三种方法实现
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;     // 滤波器(核)大小
  int stride_h_, stride_w_;     // 跨步大小
  int pad_h_, pad_w_;           // 图像填充大小    
  int channels_;                // 图像通道数
  int height_, width_;          // 图像高、宽    
  int pooled_height_, pooled_width_;
  // 池化后图像高、宽
  // pooled_height_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1
  // pooled_width_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1

  bool global_pooling_;         // 是否全区域池化(将整幅图像降采样为1*1)
  Blob<Dtype> rand_idx_;        // 随机采样索引,Pooling方法为STOCHASTIC时用到并会Reshape
  Blob<int> max_idx_;           // 最大值采样索引，Pooling方法为MAX时用到并会Reshape
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
