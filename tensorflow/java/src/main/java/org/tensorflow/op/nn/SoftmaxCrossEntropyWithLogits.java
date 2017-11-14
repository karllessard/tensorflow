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

public final class SoftmaxCrossEntropyWithLogits<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SoftmaxCrossEntropyWithLogits operation to the graph.
   * 
   * @param scope Current graph scope
   * @param features
   * @param labels
   * @return a new instance of SoftmaxCrossEntropyWithLogits
   **/
  public static <T> SoftmaxCrossEntropyWithLogits<T> create(Scope scope, Operand<T> features, Operand<T> labels) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SoftmaxCrossEntropyWithLogits", scope.makeOpName("SoftmaxCrossEntropyWithLogits"));
    opBuilder.addInput(features.asOutput());
    opBuilder.addInput(labels.asOutput());
    return new SoftmaxCrossEntropyWithLogits<T>(opBuilder.build());
  }
  
  public Output<T> loss() {
    return loss;
  }
  
  public Output<T> backprop() {
    return backprop;
  }
  
  private Output<T> loss;
  private Output<T> backprop;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SoftmaxCrossEntropyWithLogits(Operation operation) {
    super(operation);
    int outputIdx = 0;
    loss = operation.output(outputIdx++);
    backprop = operation.output(outputIdx++);
  }
}
