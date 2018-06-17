#include "caffe/util/math_functions.hpp"


namespace caffe {

/*****************************************************************
*Function:      AdaDeltaUpdate()
*Description:   梯度更新
*Calls:         
*Called By:     adadelta_update_gpu() 
*Input:         
*Output:
*Return:
*Others:        参考文献：ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
*****************************************************************/
template <typename Dtype>
__global__ void AdaDeltaUpdate(int N, Dtype* g, Dtype* h, Dtype* h2,
    Dtype momentum, Dtype delta, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = momentum * h[i] + (1-momentum) * gi * gi;
    gi = gi * sqrt((h2[i] + delta) / (hi + delta));
    h2[i] = momentum * h2[i] + (1-momentum) * gi * gi;
    g[i] = local_rate * gi;
  }
}

/*****************************************************************
*Function:      adadelta_update_gpu()
*Description:   梯度更新
*Calls:         AdaDeltaUpdate()
*Called By:     ComputeUpdateValue() 
*Input:         
*Output:
*Return:
*Others:        GPU版本
*****************************************************************/
template <typename Dtype>
void adadelta_update_gpu(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum,
    Dtype delta, Dtype local_rate) {
  AdaDeltaUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, h2, momentum, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void adadelta_update_gpu<float>(int , float*, float*, float*,
    float, float, float);
template void adadelta_update_gpu<double>(int, double*, double*, double*,
    double, double, double);

}  // namespace caffe
