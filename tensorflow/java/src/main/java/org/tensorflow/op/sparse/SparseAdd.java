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

public final class SparseAdd<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseAdd operation to the graph.
   * 
   * @param scope Current graph scope
   * @param aIndices
   * @param aValues
   * @param aShape
   * @param bIndices
   * @param bValues
   * @param bShape
   * @param thresh
   * @return a new instance of SparseAdd
   **/
  public static <T, U> SparseAdd<T> create(Scope scope, Operand<Long> aIndices, Operand<T> aValues, Operand<Long> aShape, Operand<Long> bIndices, Operand<T> bValues, Operand<Long> bShape, Operand<U> thresh) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseAdd", scope.makeOpName("SparseAdd"));
    opBuilder.addInput(aIndices.asOutput());
    opBuilder.addInput(aValues.asOutput());
    opBuilder.addInput(aShape.asOutput());
    opBuilder.addInput(bIndices.asOutput());
    opBuilder.addInput(bValues.asOutput());
    opBuilder.addInput(bShape.asOutput());
    opBuilder.addInput(thresh.asOutput());
    return new SparseAdd<T>(opBuilder.build());
  }
  
  public Output<Long> sumIndices() {
    return sumIndices;
  }
  
  public Output<T> sumValues() {
    return sumValues;
  }
  
  public Output<Long> sumShape() {
    return sumShape;
  }
  
  private Output<Long> sumIndices;
  private Output<T> sumValues;
  private Output<Long> sumShape;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseAdd(Operation operation) {
    super(operation);
    int outputIdx = 0;
    sumIndices = operation.output(outputIdx++);
    sumValues = operation.output(outputIdx++);
    sumShape = operation.output(outputIdx++);
  }
}
