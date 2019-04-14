#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

/*****************************************************************
Function:      ReLULayer<Dtype>::Forward_cpu()
*Description:  relu 激活函数  CPU实现前向传播
*Calls:        
*Called By:     
*Input:         
*Output:
*Return:
*Others:       
*****************************************************************/
template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();// leaky常数
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))            // relu
        + negative_slope * std::min(bottom_data[i], Dtype(0));  // 顺便实现leaky relu
  }
}

/*****************************************************************
Function:      ReLULayer<Dtype>::Backward_cpu()
*Description:  relu 激活函数  CPU实现反向传播
*Calls:        
*Called By:     
*Input:         
*Output:
*Return:
*Others:       
*****************************************************************/
template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
        // 导数：x>0 : 1
        //       x<=0: negative_slope
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
