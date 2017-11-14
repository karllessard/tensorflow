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
package org.tensorflow.op.parsing;

import java.util.Arrays;
import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class ParseExample extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new ParseExample operation to the graph.
   * 
   * @param scope Current graph scope
   * @param serialized
   * @param names
   * @param sparseKeys
   * @param denseKeys
   * @param denseDefaults
   * @param sparseTypes
   * @param denseShapes
   * @return a new instance of ParseExample
   **/
  public static ParseExample create(Scope scope, Operand<String> serialized, Operand<String> names, Iterable<Operand<String>> sparseKeys, Iterable<Operand<String>> denseKeys, Iterable<Operand<?>> denseDefaults, List<DataType> sparseTypes, List<Shape> denseShapes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ParseExample", scope.makeOpName("ParseExample"));
    opBuilder.addInput(serialized.asOutput());
    opBuilder.addInput(names.asOutput());
    opBuilder.addInputList(Operands.asOutputs(sparseKeys));
    opBuilder.addInputList(Operands.asOutputs(denseKeys));
    opBuilder.addInputList(Operands.asOutputs(denseDefaults));
    opBuilder.setAttr("sparseTypes", sparseTypes.toArray(new DataType[sparseTypes.size()]));
    opBuilder.setAttr("denseShapes", denseShapes.toArray(new Shape[denseShapes.size()]));
    return new ParseExample(opBuilder.build());
  }
  
  public List<Output<Long>> sparseIndices() {
    return sparseIndices;
  }
  
  public List<Output<DataType>> sparseValues() {
    return sparseValues;
  }
  
  public List<Output<Long>> sparseShapes() {
    return sparseShapes;
  }
  
  public List<Output<?>> denseValues() {
    return denseValues;
  }
  
  private List<Output<Long>> sparseIndices;
  private List<Output<DataType>> sparseValues;
  private List<Output<Long>> sparseShapes;
  private List<Output<?>> denseValues;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ParseExample(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int sparseIndicesLength = operation.outputListLength("sparseIndices");
    sparseIndices = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, sparseIndicesLength));
    outputIdx += sparseIndicesLength;
    int sparseValuesLength = operation.outputListLength("sparseValues");
    sparseValues = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, sparseValuesLength));
    outputIdx += sparseValuesLength;
    int sparseShapesLength = operation.outputListLength("sparseShapes");
    sparseShapes = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, sparseShapesLength));
    outputIdx += sparseShapesLength;
    int denseValuesLength = operation.outputListLength("denseValues");
    denseValues = Arrays.asList(operation.outputList(outputIdx, denseValuesLength));
    outputIdx += denseValuesLength;
  }
}
