/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"
#include "tensorflow/java/src/main/native/eager_constant_builder_jni.h"

namespace {

TF_Tensor* requireTensor(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kNullPointerException,
                   "close() was called on the Tensor");
    return nullptr;
  }
  return reinterpret_cast<TF_Tensor*>(handle);
}

}  // namespace

JNIEXPORT jlong JNICALL
Java_org_tensorflow_EagerConstantBuilder_allocateTensorHandle(
    JNIEnv* env, jclass clazz, jlong thandle) {
  TF_Tensor* tensor = requireTensor(env, thandle);
  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandle(tensor, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  static_assert(sizeof(jlong) >= sizeof(TFE_TensorHandle*),
                "Cannot represent a C TFE_TensorHandle as a Java long");
  return reinterpret_cast<jlong>(tensor_handle);
}
