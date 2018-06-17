#include "caffe/util/math_functions.hpp"


namespace caffe {

/*****************************************************************
*Function:      NesterovUpdate()
*Description:   计算更新值
*Calls:         
*Called By:     nesterov_update_gpu()
*Input:         
*Output:
*Return:
*Others:        .Nesterov是Momentum的变种,相当于添加了矫正因子的Momentum   參考文獻：On the importance of initialization and momentum in deep learning
*****************************************************************/
template <typename Dtype>
__global__ void NesterovUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float hi = h[i];
    float hi_new = h[i] = momentum * hi + local_rate * g[i];
    g[i] = (1+momentum) * hi_new - momentum * hi;
  }
}


/*****************************************************************
*Function:      nesterov_update_gpu()
*Description:   计算更新值
*Calls:         NesterovUpdate() 
*Called By:      
*Input:         
*Output:
*Return:
*Others:       GPU版本 
*****************************************************************/
template <typename Dtype>
void nesterov_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  NesterovUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void nesterov_update_gpu<float>(int, float*, float*, float, float);
template void nesterov_update_gpu<double>(int, double*, double*, double,
    double);

}  // namespace caffe
