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
package org.tensorflow.op.math;

import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class AddN<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new AddN operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @return a new instance of AddN
   **/
  public static <T> AddN<T> create(Scope scope, Iterable<Operand<T>> inputs) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AddN", scope.makeOpName("AddN"));
    opBuilder.addInputList(Operands.asOutputs(inputs));
    return new AddN<T>(opBuilder.build());
  }
  
  public Output<T> sum() {
    return sum;
  }
  
  @Override
  public Output<T> asOutput() {
    return sum;
  }
  
  private Output<T> sum;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private AddN(Operation operation) {
    super(operation);
    int outputIdx = 0;
    sum = operation.output(outputIdx++);
  }
}
