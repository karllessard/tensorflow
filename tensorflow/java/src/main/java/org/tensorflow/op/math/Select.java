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

public final class Select<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new Select operation to the graph.
   * 
   * @param scope Current graph scope
   * @param condition
   * @param t
   * @param e
   * @return a new instance of Select
   **/
  public static <T> Select<T> create(Scope scope, Operand<Boolean> condition, Operand<T> t, Operand<T> e) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Select", scope.makeOpName("Select"));
    opBuilder.addInput(condition.asOutput());
    opBuilder.addInput(t.asOutput());
    opBuilder.addInput(e.asOutput());
    return new Select<T>(opBuilder.build());
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
  private Select(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
