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

import java.util.Arrays;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseSplit<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SparseSplit operation to the graph.
   * 
   * @param scope Current graph scope
   * @param splitDim
   * @param indices
   * @param values
   * @param shape
   * @param numSplit
   * @return a new instance of SparseSplit
   **/
  public static <T> SparseSplit<T> create(Scope scope, Operand<Long> splitDim, Operand<Long> indices, Operand<T> values, Operand<Long> shape, Integer numSplit) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseSplit", scope.makeOpName("SparseSplit"));
    opBuilder.addInput(splitDim.asOutput());
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(values.asOutput());
    opBuilder.addInput(shape.asOutput());
    opBuilder.setAttr("numSplit", numSplit);
    return new SparseSplit<T>(opBuilder.build());
  }
  
  public List<Output<Long>> outputIndices() {
    return outputIndices;
  }
  
  public List<Output<T>> outputValues() {
    return outputValues;
  }
  
  public List<Output<Long>> outputShape() {
    return outputShape;
  }
  
  private List<Output<Long>> outputIndices;
  private List<Output<T>> outputValues;
  private List<Output<Long>> outputShape;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseSplit(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int outputIndicesLength = operation.outputListLength("outputIndices");
    outputIndices = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, outputIndicesLength));
    outputIdx += outputIndicesLength;
    int outputValuesLength = operation.outputListLength("outputValues");
    outputValues = Arrays.asList((Output<T>[])operation.outputList(outputIdx, outputValuesLength));
    outputIdx += outputValuesLength;
    int outputShapeLength = operation.outputListLength("outputShape");
    outputShape = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, outputShapeLength));
    outputIdx += outputShapeLength;
  }
}
