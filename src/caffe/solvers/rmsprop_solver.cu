#include "caffe/util/math_functions.hpp"


namespace caffe {

/*****************************************************************
*Function:      RMSPropUpdate()
*Description:   计算更新值
*Calls:         
*Called By:     prop_update_gpu()
*Input:         
*Output:
*Return:
*Others:       RMSProp增加了一个衰减系数来控制历史信息的获取多少，目的改变梯度累积为指数衰减的移动平均以丢弃遥远的过去历史 
*****************************************************************/
template <typename Dtype>
__global__ void RMSPropUpdate(int N, Dtype* g, Dtype* h,
    Dtype rms_decay, Dtype delta, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = rms_decay*h[i] + (1-rms_decay)*gi*gi;
    g[i] = local_rate * g[i] / (sqrt(hi) + delta);
  }
}

/*****************************************************************
*Function:      rmsprop_update_gpu()
*Description:   计算更新值
*Calls:         RMSPropUpdate() 
*Called By:      
*Input:         
*Output:
*Return:
*Others:       GPU版本 
*****************************************************************/
template <typename Dtype>
void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay,
    Dtype delta, Dtype local_rate) {
  RMSPropUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, rms_decay, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void rmsprop_update_gpu<float>(int, float*, float*, float, float,
    float);
template void rmsprop_update_gpu<double>(int, double*, double*, double, double,
    double);

}  // namespace caffe
