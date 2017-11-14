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
package org.tensorflow.op.sparse;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseReshape extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseReshape operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputIndices
   * @param inputShape
   * @param newShape
   * @return a new instance of SparseReshape
   **/
  public static SparseReshape create(Scope scope, Operand<Long> inputIndices, Operand<Long> inputShape, Operand<Long> newShape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseReshape", scope.makeOpName("SparseReshape"));
    opBuilder.addInput(inputIndices.asOutput());
    opBuilder.addInput(inputShape.asOutput());
    opBuilder.addInput(newShape.asOutput());
    return new SparseReshape(opBuilder.build());
  }
  
  public Output<Long> outputIndices() {
    return outputIndices;
  }
  
  public Output<Long> outputShape() {
    return outputShape;
  }
  
  private Output<Long> outputIndices;
  private Output<Long> outputShape;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseReshape(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputIndices = operation.output(outputIdx++);
    outputShape = operation.output(outputIdx++);
  }
}
