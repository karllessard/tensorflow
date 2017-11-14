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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class PlaceholderWithDefault<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new PlaceholderWithDefault operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param shape
   * @return a new instance of PlaceholderWithDefault
   **/
  public static <T> PlaceholderWithDefault<T> create(Scope scope, Operand<T> input, Shape shape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("PlaceholderWithDefault", scope.makeOpName("PlaceholderWithDefault"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("shape", shape);
    return new PlaceholderWithDefault<T>(opBuilder.build());
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
  private PlaceholderWithDefault(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
