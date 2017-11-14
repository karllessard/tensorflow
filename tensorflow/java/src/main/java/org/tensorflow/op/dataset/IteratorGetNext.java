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
package org.tensorflow.op.dataset;

import java.util.Arrays;
import java.util.Iterator;
import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class IteratorGetNext extends PrimitiveOp implements Iterable<Operand<DataType>> {
  
  /**
   * Factory method to create a class to wrap a new IteratorGetNext operation to the graph.
   * 
   * @param scope Current graph scope
   * @param iterator
   * @param outputTypes
   * @param outputShapes
   * @return a new instance of IteratorGetNext
   **/
  public static IteratorGetNext create(Scope scope, Operand<?> iterator, List<DataType> outputTypes, List<Shape> outputShapes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("IteratorGetNext", scope.makeOpName("IteratorGetNext"));
    opBuilder.addInput(iterator.asOutput());
    opBuilder.setAttr("outputTypes", outputTypes.toArray(new DataType[outputTypes.size()]));
    opBuilder.setAttr("outputShapes", outputShapes.toArray(new Shape[outputShapes.size()]));
    return new IteratorGetNext(opBuilder.build());
  }
  
  public List<Output<DataType>> components() {
    return components;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<DataType>> iterator() {
    return (Iterator) components.iterator();
  }
  
  private List<Output<DataType>> components;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private IteratorGetNext(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int componentsLength = operation.outputListLength("components");
    components = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, componentsLength));
    outputIdx += componentsLength;
  }
}
