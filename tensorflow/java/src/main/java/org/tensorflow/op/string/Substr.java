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
package org.tensorflow.op.string;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Substr extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new Substr operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param pos
   * @param len
   * @return a new instance of Substr
   **/
  public static <T> Substr create(Scope scope, Operand<String> input, Operand<T> pos, Operand<T> len) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Substr", scope.makeOpName("Substr"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(pos.asOutput());
    opBuilder.addInput(len.asOutput());
    return new Substr(opBuilder.build());
  }
  
  public Output<String> output() {
    return output;
  }
  
  @Override
  public Output<String> asOutput() {
    return output;
  }
  
  private Output<String> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Substr(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
