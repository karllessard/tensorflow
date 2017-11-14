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

public final class SparseTensorDenseAdd<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Factory method to create a class to wrap a new SparseTensorDenseAdd operation to the graph.
   * 
   * @param scope Current graph scope
   * @param aIndices
   * @param aValues
   * @param aShape
   * @param b
   * @return a new instance of SparseTensorDenseAdd
   **/
  public static <T, U> SparseTensorDenseAdd<U> create(Scope scope, Operand<T> aIndices, Operand<U> aValues, Operand<T> aShape, Operand<U> b) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseTensorDenseAdd", scope.makeOpName("SparseTensorDenseAdd"));
    opBuilder.addInput(aIndices.asOutput());
    opBuilder.addInput(aValues.asOutput());
    opBuilder.addInput(aShape.asOutput());
    opBuilder.addInput(b.asOutput());
    return new SparseTensorDenseAdd<U>(opBuilder.build());
  }
  
  public Output<U> output() {
    return output;
  }
  
  @Override
  public Output<U> asOutput() {
    return output;
  }
  
  private Output<U> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseTensorDenseAdd(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
