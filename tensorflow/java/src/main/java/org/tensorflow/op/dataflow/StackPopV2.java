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

public final class StackPopV2<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new StackPopV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param elem_type
   * @return a new instance of StackPopV2
   **/
  public static <T> StackPopV2<T> create(Scope scope, Operand<?> handle, Class<T> elem_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("StackPopV2", scope.makeOpName("StackPopV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.setAttr("elem_type", DataType.fromClass(elem_type));
    return new StackPopV2<T>(opBuilder.build());
  }
  
  public Output<T> elem() {
    return elem;
  }
  
  @Override
  public Output<T> asOutput() {
    return elem;
  }
  
  private Output<T> elem;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private StackPopV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    elem = operation.output(outputIdx++);
  }
}
