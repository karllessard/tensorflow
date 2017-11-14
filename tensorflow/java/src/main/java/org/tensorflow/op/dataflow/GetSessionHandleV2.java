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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class GetSessionHandleV2 extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Factory method to create a class to wrap a new GetSessionHandleV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param value
   * @return a new instance of GetSessionHandleV2
   **/
  public static <T> GetSessionHandleV2 create(Scope scope, Operand<T> value) {
    OperationBuilder opBuilder = scope.graph().opBuilder("GetSessionHandleV2", scope.makeOpName("GetSessionHandleV2"));
    opBuilder.addInput(value.asOutput());
    return new GetSessionHandleV2(opBuilder.build());
  }
  
  public Output<?> handle() {
    return handle;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<Object> asOutput() {
    return (Output<Object>) handle;
  }
  
  private Output<?> handle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private GetSessionHandleV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
