#include "caffe/util/math_functions.hpp"


namespace caffe {

/*****************************************************************
*Function:      AdaGradUpdate()
*Description:   计算更新值
*Calls:         
*Called By:     adagrad_update_gpu() 
*Input:         
*Output:
*Return:
*Others:        adagrad方法是将每一个参数的每一次迭代的梯度取平方累加再开方，用基础学习率除以这个数，来做学习率的动态更新  參考文獻：Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
*****************************************************************/
template <typename Dtype>
__global__ void AdaGradUpdate(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = h[i] + gi*gi;
    g[i] = local_rate * gi / (sqrt(hi) + delta);
  }
}



/*****************************************************************
*Function:      adagrad_update_gpu()
*Description:   计算更新值
*Calls:         AdaGradUpdate() 
*Called By:      
*Input:         
*Output:
*Return:
*Others:       GPU版本 
*****************************************************************/
template <typename Dtype>
void adagrad_update_gpu(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  AdaGradUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void adagrad_update_gpu<float>(int, float*, float*, float, float);
template void adagrad_update_gpu<double>(int, double*, double*, double, double);

}  // namespace caffe
