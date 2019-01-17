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

package org.tensorflow.eager;

import java.util.Stack;

import org.tensorflow.AbstractOperation;
import org.tensorflow.DataType;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

final class EagerOperation extends AbstractOperation implements AutoCloseable {
  
  EagerOperation(Tensor<?> tensor, String type, String name) {
    this(new long[] { allocateTensorHandle(tensor.getNativeHandle()) }, type, name);
    resolvedTensors[0] = tensor;
    allocatedResources.push(tensor);
  }

  EagerOperation(long[] nativeHandles, String type, String name) {
    this.nativeHandles = nativeHandles;
    this.resolvedTensors = new Tensor<?>[nativeHandles.length];
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
    return nativeHandles.length;
  }

  @Override
  public int outputListLength(final String name) {
    return 0; // TODO
  }

  @Override
  public Output<?>[] outputList(int idx, int length) {
    Output<?>[] outputs = new Output<?>[length];
    for (int i = 0; i < length; ++i) {
      outputs[i] = output(idx + i);
    }
    return outputs;
  }

  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public <T> Output<T> output(int idx) {
    return new Output(this, idx);
  }

  @Override
  public int inputListLength(final String name) {
    return 0; // TODO
  }

  @Override
  public long getUnsafeNativeHandle(int outputIndex) {
    return nativeHandles[outputIndex];
  }

  @Override
  public long[] shape(int outputIndex) {
    if (resolvedTensors[outputIndex] != null) {
      return resolvedTensors[outputIndex].shape();
    }
    long nativeHandle = nativeHandles[outputIndex];
    long[] shape = new long[numDims(nativeHandle)];
    for (int i = 0; i < shape.length; ++i) {
      shape[i] = dim(nativeHandle, i);
    }
    return shape;
  }

  @Override
  public DataType dtype(int outputIndex) {
    if (resolvedTensors[outputIndex] != null) {
      return resolvedTensors[outputIndex].dataType();
    }
    return DataType.fromC(dataType(nativeHandles[outputIndex]));
  }
  
  @Override
  public Tensor<?> outputTensor(int outputIndex) {
    if (resolvedTensors[outputIndex] != null) {
      return resolvedTensors[outputIndex];
    }
    long handle = resolveTensorHandle(nativeHandles[outputIndex]);
    Tensor<?> tensor = Tensor.fromHandle(handle);
    resolvedTensors[outputIndex] = tensor;
    session.collector().add(tensor);
    return tensor;
  }

  @Override
  public void release() {
    for (int i = 0; i < nativeHandles.length; ++i) {
      deleteTensorHandle(nativeHandles[i]);
    }
  }

  private final long[] nativeHandles;
  private final Tensor<?>[] resolvedTensors;
  private final String type;
  private final String name;
  private final Stack<AutoCloseable> allocatedResources = new Stack<>();

  private static native long allocateTensorHandle(long thandle);

  private static native void deleteTensorHandle(long handle);

  private static native long resolveTensorHandle(long handle);

  private static native int dataType(long handle);

  private static native int numDims(long handle);

  private static native long dim(long handle, int index);
}
