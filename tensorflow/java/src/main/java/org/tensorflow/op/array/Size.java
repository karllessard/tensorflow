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
package org.tensorflow.op.array;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Size<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Factory method to create a class to wrap a new Size operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param out_type
   * @return a new instance of Size
   **/
  public static <T, U> Size<U> create(Scope scope, Operand<T> input, Class<U> out_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Size", scope.makeOpName("Size"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    return new Size<U>(opBuilder.build());
  }
  
  public Output<U> output() {
    return output;
  }
  
  @Override
  public Output<U> asOutput() {
    return output;
  }
  
  private Output<U> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Size(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
