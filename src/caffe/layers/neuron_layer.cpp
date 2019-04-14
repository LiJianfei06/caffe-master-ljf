#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/*****************************************************************
Function:      NeuronLayer<Dtype>::Reshape()
*Description:  功能是将输出blob的形状改为和输入blob一样
*Calls:        
*Called By:     
*Input:         
*Output:
*Return:
*Others:       
*****************************************************************/
template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
