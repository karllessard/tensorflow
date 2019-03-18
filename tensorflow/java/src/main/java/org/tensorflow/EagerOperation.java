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

import java.io.Closeable;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Implementation for an {@link Operation} executed eagerly.
 *
 * <p>EagerOperation resources are automatically released when the {@link EagerSession} they are a
 * part of is closed. Thus, if {@link EagerSession#close()} has been invoked, then methods on the 
 * EagerOperation instance may fail with an {@code IllegalStateException}.
 *
 * <p>As opposed to {@link GraphOperation}, the application is responsible to make sure that no other
 * thread is using an EagerOperation while its {@link EagerSession} is being closed. This is to
 * improve performances. Other than that, EagerOperation instances are immutable and thread-safe.
 */
class EagerOperation extends AbstractOperation implements Closeable {
  
  EagerOperation(long nativeHandle, long[] outputNativeHandles, String type, String name) {
    this.nativeHandle = nativeHandle;
    this.outputNativeHandles = outputNativeHandles;
    this.outputTensors = new AtomicReferenceArray<Tensor<?>>(outputNativeHandles.length);
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
    return outputListLength(getNativeHandle(), name);
  }

  @Override
  public int inputListLength(final String name) {
    return inputListLength(getNativeHandle(), name);
  }

  @Override
  public long getUnsafeNativeHandle(int outputIndex) {
    if (nativeHandle == 0L) {
      throw new IllegalStateException("This operation has been closed");
    }
    return outputNativeHandles[outputIndex];
  }

  @Override
  public long[] shape(int outputIndex) {
    Tensor<?> tensor = outputTensors.get(outputIndex);
    if (tensor != null) {
      return tensor.shape();
    }
    long outputNativeHandle = getNativeHandle(outputIndex);
    long[] shape = new long[numDims(outputNativeHandle)];
    for (int i = 0; i < shape.length; ++i) {
      shape[i] = dim(outputNativeHandle, i);
    }
    return shape;
  }

  @Override
  public DataType dtype(int outputIndex) {
    Tensor<?> tensor = outputTensors.get(outputIndex);
    if (tensor != null) {
      return tensor.dataType();
    }
    return DataType.fromC(dataType(getNativeHandle(outputIndex)));
  }
  
  @Override
  public Tensor<?> tensor(int outputIndex) {
    Tensor<?> tensor = outputTensors.get(outputIndex);
    if (tensor == null) {
      // Take an optimistic approach, where we attempt to resolve the output tensor without locking.
      // If another thread resolved it meanwhile, release our copy and use the existing one.
      tensor = Tensor.fromHandle(resolveTensorHandle(getNativeHandle(outputIndex)));
      if (!outputTensors.compareAndSet(outputIndex, null, tensor)) {
        tensor.close();
        tensor = outputTensors.get(outputIndex);
      }
    }
    return tensor;
  }

  @Override
  public void close() {
    // Release first our operation handle so we can start returning clean exceptions in case other methods
    // are being called by other threads (should be prevented by the application, though).
    long tmpNativeHandle = nativeHandle;
    nativeHandle = 0L;
    delete(tmpNativeHandle);

    for (int i = 0; i < outputTensors.length(); ++i) {
      Tensor<?> tensor = outputTensors.get(i);
      if (tensor != null) {
        tensor.close();
        tensor = null;
      }
    }
    for (int i = 0; i < outputNativeHandles.length; ++i) {
      deleteTensorHandle(outputNativeHandles[i]);
      outputNativeHandles[i] = 0L;
    }
  }

  private long nativeHandle;
  private final long[] outputNativeHandles;  // all values are set to a valid native tensor handle
  private final AtomicReferenceArray<Tensor<?>> outputTensors;  // only tensors that has been accessed so far are non-null
  private final String type;
  private final String name;
  
  private long getNativeHandle() {
    if (nativeHandle == 0L) {
      throw new IllegalStateException("This operation has been closed");
    }
    return nativeHandle;
  }
  
  private static native void delete(long handle);

  private static native long allocateTensorHandle(long thandle);

  private static native void deleteTensorHandle(long handle);

  private static native long resolveTensorHandle(long handle);
  
  private static native int outputListLength(long handle, String name);

  private static native int inputListLength(long handle, String name);

  private static native int dataType(long handle);

  private static native int numDims(long handle);

  private static native long dim(long handle, int index);
}
