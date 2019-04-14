#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/*****************************************************************
Function:      sigmoid()
*Description:  sigmoid 激活函数  
*Calls:
*Called By:    SigmoidLayer<Dtype>::Forward_cpu() 
*Input:         
*Output:
*Return:
*Others:       利用 tanh()  函数间接实现 双曲正切函数 在cmath里
*****************************************************************/
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

/*****************************************************************
Function:      SigmoidLayer<Dtype>::Forward_cpu()
*Description:  CPU 实现 sigmoid 激活函数前向传播 
*Calls:        sigmoid()
*Called By:     
*Input:         
*Output:
*Return:
*Others:       
*****************************************************************/
template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

/*****************************************************************
Function:      SigmoidLayer<Dtype>::Backward_cpu()
*Description:  CPU 实现 sigmoid 激活函数反向传播 
*Calls:        
*Called By:     
*Input:         
*Output:
*Return:
*Others:       
*****************************************************************/
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);  // 求导就是了
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
