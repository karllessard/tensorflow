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
package org.tensorflow.op.state;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class IsVariableInitialized extends PrimitiveOp implements Operand<Boolean> {
  
  /**
   * Factory method to create a class to wrap a new IsVariableInitialized operation to the graph.
   * 
   * @param scope Current graph scope
   * @param ref
   * @return a new instance of IsVariableInitialized
   **/
  public static <T> IsVariableInitialized create(Scope scope, Operand<T> ref) {
    OperationBuilder opBuilder = scope.graph().opBuilder("IsVariableInitialized", scope.makeOpName("IsVariableInitialized"));
    opBuilder.addInput(ref.asOutput());
    return new IsVariableInitialized(opBuilder.build());
  }
  
  public Output<Boolean> isInitialized() {
    return isInitialized;
  }
  
  @Override
  public Output<Boolean> asOutput() {
    return isInitialized;
  }
  
  private Output<Boolean> isInitialized;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private IsVariableInitialized(Operation operation) {
    super(operation);
    int outputIdx = 0;
    isInitialized = operation.output(outputIdx++);
  }
}
