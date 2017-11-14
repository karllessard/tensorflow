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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseCross<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseCross operation to the graph.
   * 
   * @param scope Current graph scope
   * @param indices
   * @param values
   * @param shapes
   * @param denseInputs
   * @param hashedOutput
   * @param numBuckets
   * @param hashKey
   * @param out_type
   * @param internal_type
   * @return a new instance of SparseCross
   **/
  public static <T, U> SparseCross<T> create(Scope scope, Iterable<Operand<Long>> indices, Iterable<Operand<?>> values, Iterable<Operand<Long>> shapes, Iterable<Operand<?>> denseInputs, Boolean hashedOutput, Integer numBuckets, Integer hashKey, Class<T> out_type, Class<U> internal_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseCross", scope.makeOpName("SparseCross"));
    opBuilder.addInputList(Operands.asOutputs(indices));
    opBuilder.addInputList(Operands.asOutputs(values));
    opBuilder.addInputList(Operands.asOutputs(shapes));
    opBuilder.addInputList(Operands.asOutputs(denseInputs));
    opBuilder.setAttr("hashedOutput", hashedOutput);
    opBuilder.setAttr("numBuckets", numBuckets);
    opBuilder.setAttr("hashKey", hashKey);
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    opBuilder.setAttr("internal_type", DataType.fromClass(internal_type));
    return new SparseCross<T>(opBuilder.build());
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
  private SparseCross(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputIndices = operation.output(outputIdx++);
    outputValues = operation.output(outputIdx++);
    outputShape = operation.output(outputIdx++);
  }
}
