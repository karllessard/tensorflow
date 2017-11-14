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
package org.tensorflow.op.dataflow;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseAccumulatorApplyGradient extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseAccumulatorApplyGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param localStep
   * @param gradientIndices
   * @param gradientValues
   * @param gradientShape
   * @param hasKnownShape
   * @return a new instance of SparseAccumulatorApplyGradient
   **/
  public static <T> SparseAccumulatorApplyGradient create(Scope scope, Operand<String> handle, Operand<Long> localStep, Operand<Long> gradientIndices, Operand<T> gradientValues, Operand<Long> gradientShape, Boolean hasKnownShape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseAccumulatorApplyGradient", scope.makeOpName("SparseAccumulatorApplyGradient"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(localStep.asOutput());
    opBuilder.addInput(gradientIndices.asOutput());
    opBuilder.addInput(gradientValues.asOutput());
    opBuilder.addInput(gradientShape.asOutput());
    opBuilder.setAttr("hasKnownShape", hasKnownShape);
    return new SparseAccumulatorApplyGradient(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseAccumulatorApplyGradient(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
