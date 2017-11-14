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
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Merge<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new Merge operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @return a new instance of Merge
   **/
  public static <T> Merge<T> create(Scope scope, Iterable<Operand<T>> inputs) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Merge", scope.makeOpName("Merge"));
    opBuilder.addInputList(Operands.asOutputs(inputs));
    return new Merge<T>(opBuilder.build());
  }
  
  public Output<T> output() {
    return output;
  }
  
  public Output<Integer> valueIndex() {
    return valueIndex;
  }
  
  private Output<T> output;
  private Output<Integer> valueIndex;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Merge(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    valueIndex = operation.output(outputIdx++);
  }
}
