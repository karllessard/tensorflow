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
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Range<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new Range operation to the graph.
   * 
   * @param scope Current graph scope
   * @param start
   * @param limit
   * @param delta
   * @return a new instance of Range
   **/
  public static <T> Range<T> create(Scope scope, Operand<T> start, Operand<T> limit, Operand<T> delta) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Range", scope.makeOpName("Range"));
    opBuilder.addInput(start.asOutput());
    opBuilder.addInput(limit.asOutput());
    opBuilder.addInput(delta.asOutput());
    return new Range<T>(opBuilder.build());
  }
  
  public Output<T> output() {
    return output;
  }
  
  @Override
  public Output<T> asOutput() {
    return output;
  }
  
  private Output<T> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Range(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
