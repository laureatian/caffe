#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (M_ == 1) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype) 1., weight,
                            bottom_data, (Dtype) 0., top_data);
      if (bias_term_)
        caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                              this->blobs_[1]->gpu_data(), top_data);
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,
                            transpose_ ? CblasNoTrans : CblasTrans,
                            M_, N_, K_, (Dtype) 1.,
                            bottom_data, weight, (Dtype) 0., top_data);
      if (bias_term_)
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                              bias_multiplier_.gpu_data(),
                              this->blobs_[1]->gpu_data(), (Dtype) 1.,
                              top_data);
    }
#endif  // USE CUDA
  } else {
#ifdef USE_GREENTEA
    if (M_ == 1) {
//#define VECMUL_PROFILE
#ifdef VECMUL_PROFILE
      Timer timer;
      timer.initted();
      timer.Start();
      for(uint i = 0; i < 100; ++i) {
#endif
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
        const viennacl::ocl::device &device = ctx.current_device();
        if (device.vendor().find("Intel") != std::string::npos) {
            viennacl::ocl::program &program = ctx.get_program("kernel_program");
            viennacl::ocl::kernel &k = program.get_kernel("vec_mul4");
            uint M = N_;
            uint N = K_;
            size_t localsize = 128;
            size_t globalsize = M/8* localsize;

            uint argId = 0;
            k.arg(argId++, WrapHandle((cl_mem)weight, &ctx));
            k.arg(argId++, cl_uint(M));
            k.arg(argId++, cl_uint(N));
            k.arg(argId++, WrapHandle((cl_mem) bottom_data, &ctx));
            k.arg(argId++, cl_uint(N));
            k.arg(argId++, WrapHandle((cl_mem) top_data, &ctx));
            k.arg(argId++, cl_uint(M));
            k.arg(argId++, viennacl::ocl::local_mem(sizeof(cl_float8) * localsize));

            clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                         k.handle().get(), 1,
                                         NULL,
                                         &globalsize,
                                         &localsize, 0, NULL,
                                         NULL);
           clFinish(ctx.get_queue().handle().get());
           //std::cout <<"error code: " << err << std::endl;
        } else {
        greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, N_,
                               K_, (Dtype) 1., (cl_mem) weight, 0,
                               (cl_mem) bottom_data, 0, (Dtype) 0.,
                               (cl_mem) top_data, 0);
        }
#ifdef VECMUL_PROFILE
      }
      timer.Stop();
      float elapsedTime = timer.MilliSeconds();
      std::cout << "GEMV Time is: " << elapsedTime / 100.f <<" ms" << std::endl;
#endif
      if (bias_term_)
        greentea_gpu_axpy<Dtype>(this->device_->id(), N_,
                                 bias_multiplier_.cpu_data()[0],
                                 (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                 (cl_mem) top_data, 0);
    } else {
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               transpose_ ? CblasNoTrans : CblasTrans,
                               M_, N_, K_, (Dtype) 1.,
                               (cl_mem) bottom_data, 0, (cl_mem) weight, 0,
                               (Dtype) 0., (cl_mem) top_data, 0);
      if (bias_term_)
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                 (cl_mem) (bias_multiplier_.gpu_data()), 0,
                                 (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                 (Dtype) 1., (cl_mem) top_data, 0);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Gradient with respect to weight
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                              (Dtype) 1.,
                              bottom_data, top_diff, (Dtype) 1.,
                              this->blobs_[0]->mutable_gpu_diff());
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                              (Dtype) 1.,
                              top_diff, bottom_data, (Dtype) 1.,
                              this->blobs_[0]->mutable_gpu_diff());
      }
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1., top_diff,
                            bias_multiplier_.gpu_data(), (Dtype) 1.,
                            this->blobs_[1]->mutable_gpu_diff());
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff());
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff());
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Gradient with respect to weight
      if (transpose_) {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasTrans, CblasNoTrans,
                                 K_, N_, M_, (Dtype) 1., (cl_mem) bottom_data,
                                 0, (cl_mem) top_diff, 0, (Dtype) 1.,
                                 (cl_mem) (this->blobs_[0]->mutable_gpu_diff()),
                                 0);
      } else {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasTrans, CblasNoTrans,
                                 N_, K_, M_, (Dtype) 1., (cl_mem) top_diff, 0,
                                 (cl_mem) bottom_data, 0, (Dtype) 1.,
                                 (cl_mem) (this->blobs_[0]->mutable_gpu_diff()),
                                 0);
      }
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, M_, N_,
                               (Dtype) 1., (cl_mem) top_diff, 0,
                               (cl_mem) (bias_multiplier_.gpu_data()), 0,
                               (Dtype) 1.,
                               (cl_mem) (this->blobs_[1]->mutable_gpu_diff()),
                               0);
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      if (transpose_) {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasTrans, M_, K_, N_, (Dtype) 1.,
                                 (cl_mem) top_diff, 0,
                                 (cl_mem) (this->blobs_[0]->gpu_data()), 0,
                                 (Dtype) 0.,
                                 (cl_mem) (bottom[0]->mutable_gpu_diff()), 0);
      } else {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasNoTrans, M_, K_, N_, (Dtype) 1.,
                                 (cl_mem) top_diff, 0,
                                 (cl_mem) (this->blobs_[0]->gpu_data()), 0,
                                 (Dtype) 0.,
                                 (cl_mem) (bottom[0]->mutable_gpu_diff()), 0);
      }
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
