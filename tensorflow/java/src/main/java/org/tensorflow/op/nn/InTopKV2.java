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
package org.tensorflow.op.nn;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class InTopKV2 extends PrimitiveOp implements Operand<Boolean> {
  
  /**
   * Factory method to create a class to wrap a new InTopKV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param predictions
   * @param targets
   * @param k
   * @return a new instance of InTopKV2
   **/
  public static <T> InTopKV2 create(Scope scope, Operand<Float> predictions, Operand<T> targets, Operand<T> k) {
    OperationBuilder opBuilder = scope.graph().opBuilder("InTopKV2", scope.makeOpName("InTopKV2"));
    opBuilder.addInput(predictions.asOutput());
    opBuilder.addInput(targets.asOutput());
    opBuilder.addInput(k.asOutput());
    return new InTopKV2(opBuilder.build());
  }
  
  public Output<Boolean> precision() {
    return precision;
  }
  
  @Override
  public Output<Boolean> asOutput() {
    return precision;
  }
  
  private Output<Boolean> precision;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private InTopKV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    precision = operation.output(outputIdx++);
  }
}
