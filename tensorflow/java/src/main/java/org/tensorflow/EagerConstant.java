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

package org.tensorflow;

/**
 * Implementation for an {@link Operation} executed eagerly.
 */
class EagerConstant extends AbstractOperation {
  
  EagerConstant(EagerSession session, Tensor<?> tensor, long tensorHandle, String name) {
    this.nativeRef = new NativeReference(session, this, tensorHandle);
    this.tensor = tensor;
    this.name = name;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String type() {
    return "Const";
  }

  @Override
  public int numOutputs() {
    return 1;
  }

  @Override
  public int outputListLength(final String name) {
    if (!"output".equals(name)) {
      throw new IllegalArgumentException("Output \"" + name + "\" not found");
    }
    return 1;
  }

  @Override
  public int inputListLength(final String name) {
    throw new IllegalArgumentException("Input \"" + name + "\" not found");
  }

  @Override
  public long getUnsafeNativeHandle(int outputIdx) {
    if (outputIdx > 0) {
      throw new IndexOutOfBoundsException("Index " + outputIdx + " is not a valid output");
    }
    return nativeRef.tensorHandle;
  }

  @Override
  public long[] shape(int outputIndex) {
    return tensor.shape();
  }

  @Override
  public DataType dtype(int outputIndex) {
    return tensor.dataType();
  }
  
  @Override
  public Tensor<?> tensor(int outputIndex) {
    return tensor;
  }

  private static class NativeReference extends EagerSession.NativeReference {

    NativeReference(EagerSession session, EagerConstant constant, long tensorHandle) {
      super(session, constant);
      this.tensorHandle = tensorHandle;
    }

    @Override
    void delete() {
      EagerConstant.deleteTensorHandle(tensorHandle);
    }
    
    private final long tensorHandle;
  }

  private final String name;
  private final Tensor<?> tensor;
  private final NativeReference nativeRef;
  
  private static native void deleteTensorHandle(long handle);
}
