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

public final class BiasAddV1<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new BiasAddV1 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param value
   * @param bias
   * @return a new instance of BiasAddV1
   **/
  public static <T> BiasAddV1<T> create(Scope scope, Operand<T> value, Operand<T> bias) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BiasAddV1", scope.makeOpName("BiasAddV1"));
    opBuilder.addInput(value.asOutput());
    opBuilder.addInput(bias.asOutput());
    return new BiasAddV1<T>(opBuilder.build());
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
  private BiasAddV1(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
