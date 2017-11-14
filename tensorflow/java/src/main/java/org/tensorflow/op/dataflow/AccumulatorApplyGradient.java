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
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class AccumulatorApplyGradient extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new AccumulatorApplyGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param localStep
   * @param gradient
   * @return a new instance of AccumulatorApplyGradient
   **/
  public static <T> AccumulatorApplyGradient create(Scope scope, Operand<String> handle, Operand<Long> localStep, Operand<T> gradient) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AccumulatorApplyGradient", scope.makeOpName("AccumulatorApplyGradient"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(localStep.asOutput());
    opBuilder.addInput(gradient.asOutput());
    return new AccumulatorApplyGradient(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private AccumulatorApplyGradient(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
