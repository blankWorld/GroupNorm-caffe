#include <algorithm>
#include <vector>

#include "caffe/layers/group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GroupNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GroupNormParameter param = this->layer_param_.group_norm_param();
  group_num_ = param.group_num();
  eps_ = param.eps();
  
  channels_ = bottom[0]->shape(1);
  chip_num_ = int(channels_ / group_num_);

  num_ = bottom[0]->shape(0);
  CHECK_EQ(channels_ % group_num_,0);
   
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  top[0]->ReshapeLike(*bottom[0]);
 
  vector<int> sz;
  sz.push_back(num_ * group_num_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  

  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);

  spatial_dim_ = int( bottom[0]->count()/(channels_*num_) );
  
  sz[0] = spatial_dim_ * chip_num_;
  cube_sum_multiplier_.Reshape(sz);
  Dtype* cube_sum_multiplier_data = cube_sum_multiplier_.mutable_cpu_data();
  caffe_set(cube_sum_multiplier_.count(), Dtype(1.), cube_sum_multiplier_data);
  
}


#ifdef CPU_ONLY
STUB_GPU(GroupNormLayer);
#endif

INSTANTIATE_CLASS(GroupNormLayer);
REGISTER_LAYER_CLASS(GroupNorm);
}  // namespace caffe