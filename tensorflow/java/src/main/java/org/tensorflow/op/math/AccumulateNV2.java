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
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class AccumulateNV2<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new AccumulateNV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @param shape
   * @return a new instance of AccumulateNV2
   **/
  public static <T> AccumulateNV2<T> create(Scope scope, Iterable<Operand<T>> inputs, Shape shape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AccumulateNV2", scope.makeOpName("AccumulateNV2"));
    opBuilder.addInputList(Operands.asOutputs(inputs));
    opBuilder.setAttr("shape", shape);
    return new AccumulateNV2<T>(opBuilder.build());
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
  private AccumulateNV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    sum = operation.output(outputIdx++);
  }
}
