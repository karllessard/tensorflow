/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
package org.tensorflow.op.dataflow;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class GetSessionTensor<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new GetSessionTensor operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param dtype
   * @return a new instance of GetSessionTensor
   **/
  public static <T> GetSessionTensor<T> create(Scope scope, Operand<String> handle, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("GetSessionTensor", scope.makeOpName("GetSessionTensor"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new GetSessionTensor<T>(opBuilder.build());
  }
  
  public Output<T> value() {
    return value;
  }
  
  @Override
  public Output<T> asOutput() {
    return value;
  }
  
  private Output<T> value;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private GetSessionTensor(Operation operation) {
    super(operation);
    int outputIdx = 0;
    value = operation.output(outputIdx++);
  }
}
