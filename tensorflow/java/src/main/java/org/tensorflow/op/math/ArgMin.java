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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ArgMin<V> extends PrimitiveOp implements Operand<V> {
  
  /**
   * Factory method to create a class to wrap a new ArgMin operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param dimension
   * @param output_type
   * @return a new instance of ArgMin
   **/
  public static <T, U, V> ArgMin<V> create(Scope scope, Operand<T> input, Operand<U> dimension, Class<V> output_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ArgMin", scope.makeOpName("ArgMin"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(dimension.asOutput());
    opBuilder.setAttr("output_type", DataType.fromClass(output_type));
    return new ArgMin<V>(opBuilder.build());
  }
  
  public Output<V> output() {
    return output;
  }
  
  @Override
  public Output<V> asOutput() {
    return output;
  }
  
  private Output<V> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ArgMin(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
