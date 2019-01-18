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

final class EagerOperation extends AbstractOperation implements AutoCloseable {
  
  EagerOperation(Tensor<?> outputTensor, String type, String name) {
    this(new long[] { allocateTensorHandle(outputTensor.getNativeHandle()) }, type, name);
    outputTensors[0] = outputTensor;
  }

  EagerOperation(long[] outputNativeHandles, String type, String name) {
    this.outputNativeHandles = outputNativeHandles;
    this.outputTensors = new Tensor<?>[outputNativeHandles.length];
    this.type = type;
    this.name = name;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String type() {
    return type;
  }

  @Override
  public int numOutputs() {
    return outputNativeHandles.length;
  }

  @Override
  public int outputListLength(final String name) {
    long outputNativeHandle = outputNativeHandles[outputIndex];
    long[] shape = new long[numDims(outputNativeHandle)];
    for (int i = 0; i < shape.length; ++i) {
      shape[i] = dim(outputNativeHandle, i);
    }
    return shape;
    return 0; // TODO
  }

  @Override
  public int inputListLength(final String name) {
    return 0; // TODO
  }

  @Override
  public long getUnsafeNativeHandle(int outputIndex) {
    return outputNativeHandles[outputIndex];
  }

  @Override
  public long[] shape(int outputIndex) {
    if (outputTensors[outputIndex] != null) {
      return outputTensors[outputIndex].shape();
    }
    long outputNativeHandle = outputNativeHandles[outputIndex];
    long[] shape = new long[numDims(outputNativeHandle)];
    for (int i = 0; i < shape.length; ++i) {
      shape[i] = dim(outputNativeHandle, i);
    }
    return shape;
  }

  @Override
  public DataType dtype(int outputIndex) {
    if (outputTensors[outputIndex] != null) {
      return outputTensors[outputIndex].dataType();
    }
    return DataType.fromC(dataType(outputNativeHandles[outputIndex]));
  }
  
  @Override
  public Tensor<?> tensor(int outputIndex) {
    Tensor<?> tensor = outputTensors[outputIndex];
    if (tensor == null) {
      tensor = Tensor.fromHandle(resolveTensorHandle(outputNativeHandles[outputIndex]));
      outputTensors[outputIndex] = tensor;
    }
    return tensor;
  }

  @Override
  public void close() throws Exception {
    for (int i = 0; i < outputTensors.length; ++i) {
      if (outputTensors[i] != null) {
        outputTensors[i].close();
        outputTensors[i] = null;
      }
    }
    for (int i = 0; i < outputNativeHandles.length; ++i) {
      deleteTensorHandle(outputNativeHandles[i]);
      outputNativeHandles[i] = 0L;
    }
  }

  private final long[] outputNativeHandles;  // all values are set
  private final Tensor<?>[] outputTensors;  // some values might be null 
  private final String type;
  private final String name;

  private static native long allocateTensorHandle(long thandle);

  private static native void deleteTensorHandle(long handle);

  private static native long resolveTensorHandle(long handle);

  private static native int dataType(long handle);

  private static native int numDims(long handle);

  private static native long dim(long handle, int index);
}
