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

public final class Elu<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new Elu operation to the graph.
   * 
   * @param scope Current graph scope
   * @param features
   * @return a new instance of Elu
   **/
  public static <T> Elu<T> create(Scope scope, Operand<T> features) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Elu", scope.makeOpName("Elu"));
    opBuilder.addInput(features.asOutput());
    return new Elu<T>(opBuilder.build());
  }
  
  public Output<T> activations() {
    return activations;
  }
  
  @Override
  public Output<T> asOutput() {
    return activations;
  }
  
  private Output<T> activations;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Elu(Operation operation) {
    super(operation);
    int outputIdx = 0;
    activations = operation.output(outputIdx++);
  }
}
