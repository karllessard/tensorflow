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

public final class AccumulatorTakeGradient<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new AccumulatorTakeGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param numRequired
   * @param dtype
   * @return a new instance of AccumulatorTakeGradient
   **/
  public static <T> AccumulatorTakeGradient<T> create(Scope scope, Operand<String> handle, Operand<Integer> numRequired, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AccumulatorTakeGradient", scope.makeOpName("AccumulatorTakeGradient"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(numRequired.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new AccumulatorTakeGradient<T>(opBuilder.build());
  }
  
  public Output<T> average() {
    return average;
  }
  
  @Override
  public Output<T> asOutput() {
    return average;
  }
  
  private Output<T> average;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private AccumulatorTakeGradient(Operation operation) {
    super(operation);
    int outputIdx = 0;
    average = operation.output(outputIdx++);
  }
}
