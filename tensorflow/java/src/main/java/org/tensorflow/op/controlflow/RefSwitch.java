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
package org.tensorflow.op.controlflow;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class RefSwitch<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new RefSwitch operation to the graph.
   * 
   * @param scope Current graph scope
   * @param data
   * @param pred
   * @return a new instance of RefSwitch
   **/
  public static <T> RefSwitch<T> create(Scope scope, Operand<T> data, Operand<Boolean> pred) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RefSwitch", scope.makeOpName("RefSwitch"));
    opBuilder.addInput(data.asOutput());
    opBuilder.addInput(pred.asOutput());
    return new RefSwitch<T>(opBuilder.build());
  }
  
  public Output<T> outputFalse() {
    return outputFalse;
  }
  
  public Output<T> outputTrue() {
    return outputTrue;
  }
  
  private Output<T> outputFalse;
  private Output<T> outputTrue;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private RefSwitch(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputFalse = operation.output(outputIdx++);
    outputTrue = operation.output(outputIdx++);
  }
}
