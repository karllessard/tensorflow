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
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseConcat<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseConcat operation to the graph.
   * 
   * @param scope Current graph scope
   * @param indices
   * @param values
   * @param shapes
   * @param concatDim
   * @return a new instance of SparseConcat
   **/
  public static <T> SparseConcat<T> create(Scope scope, Iterable<Operand<Long>> indices, Iterable<Operand<T>> values, Iterable<Operand<Long>> shapes, Integer concatDim) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseConcat", scope.makeOpName("SparseConcat"));
    opBuilder.addInputList(Operands.asOutputs(indices));
    opBuilder.addInputList(Operands.asOutputs(values));
    opBuilder.addInputList(Operands.asOutputs(shapes));
    opBuilder.setAttr("concatDim", concatDim);
    return new SparseConcat<T>(opBuilder.build());
  }
  
  public Output<Long> outputIndices() {
    return outputIndices;
  }
  
  public Output<T> outputValues() {
    return outputValues;
  }
  
  public Output<Long> outputShape() {
    return outputShape;
  }
  
  private Output<Long> outputIndices;
  private Output<T> outputValues;
  private Output<Long> outputShape;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseConcat(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputIndices = operation.output(outputIdx++);
    outputValues = operation.output(outputIdx++);
    outputShape = operation.output(outputIdx++);
  }
}
