/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

import org.tensorflow.op.NativeOperation;

/**
 * Base class for {@link NativeOperation} implementations.
 * 
 * <p>This class is package private and therefore its usage is limited to internal 
 * purposes only.
 */
abstract class AbstractOperation implements NativeOperation {
  
  AbstractOperation(long unsafeNativeHandle) {
    this.unsafeNativeHandle = unsafeNativeHandle;
  }

  @Override
  public String toString() {
    return String.format("<%s '%s'>", type(), name());
  }

  /** 
   * Returns the native handle of the {@code outputIdx}th output of this operation.
   * 
   * <p>The nature of the returned value varies depending on the execution mode of the operation.
   * <ul>
   * <li>In eager mode, the native handle of the tensor exposed by this output is returned.</li>
   * <li>In graph mode, the native handle of the operation itself is returned (in such case, 
   * the output index also needs to be provided with the handle to the native layer)</li>
   * </ul>
   * 
   * @param outputIdx index of the output in this operation
   * @return native handle
   */
  abstract long getUnsafeNativeHandle(int outputIdx);

  /** 
   * Returns the shape of the tensor of the {code outputIdx}th output of this operation.
   * 
   * @param outputIdx index of the output of this operation
   * @return output tensor shape
   */
  abstract long[] shape(int outputIdx);

  /** 
   * Returns the datatype of the tensor of the {code outputIdx}th output of this operation.
   * 
   * @param outputIdx index of the output of this operation
   * @return output tensor datatype
   */
  abstract DataType dtype(int outputIdx);

  long getUnsafeNativeHandle() {
    return unsafeNativeHandle;
  }
  
  private final long unsafeNativeHandle;
}
