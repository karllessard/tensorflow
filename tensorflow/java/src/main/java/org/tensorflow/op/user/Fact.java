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
package org.tensorflow.op.user;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Fact extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new Fact operation to the graph.
   * 
   * @param scope Current graph scope
   * @return a new instance of Fact
   **/
  public static Fact create(Scope scope) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Fact", scope.makeOpName("Fact"));
    return new Fact(opBuilder.build());
  }
  
  public Output<String> fact() {
    return fact;
  }
  
  @Override
  public Output<String> asOutput() {
    return fact;
  }
  
  private Output<String> fact;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Fact(Operation operation) {
    super(operation);
    int outputIdx = 0;
    fact = operation.output(outputIdx++);
  }
}
