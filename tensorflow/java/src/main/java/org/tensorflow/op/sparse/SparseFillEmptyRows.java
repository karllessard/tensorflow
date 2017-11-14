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

public final class SparseFillEmptyRows<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseFillEmptyRows operation to the graph.
   * 
   * @param scope Current graph scope
   * @param indices
   * @param values
   * @param denseShape
   * @param defaultValue
   * @return a new instance of SparseFillEmptyRows
   **/
  public static <T> SparseFillEmptyRows<T> create(Scope scope, Operand<Long> indices, Operand<T> values, Operand<Long> denseShape, Operand<T> defaultValue) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseFillEmptyRows", scope.makeOpName("SparseFillEmptyRows"));
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(values.asOutput());
    opBuilder.addInput(denseShape.asOutput());
    opBuilder.addInput(defaultValue.asOutput());
    return new SparseFillEmptyRows<T>(opBuilder.build());
  }
  
  public Output<Long> outputIndices() {
    return outputIndices;
  }
  
  public Output<T> outputValues() {
    return outputValues;
  }
  
  public Output<Boolean> emptyRowIndicator() {
    return emptyRowIndicator;
  }
  
  public Output<Long> reverseIndexMap() {
    return reverseIndexMap;
  }
  
  private Output<Long> outputIndices;
  private Output<T> outputValues;
  private Output<Boolean> emptyRowIndicator;
  private Output<Long> reverseIndexMap;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseFillEmptyRows(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputIndices = operation.output(outputIdx++);
    outputValues = operation.output(outputIdx++);
    emptyRowIndicator = operation.output(outputIdx++);
    reverseIndexMap = operation.output(outputIdx++);
  }
}
