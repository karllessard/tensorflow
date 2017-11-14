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
package org.tensorflow.op.array;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ConcatOffset extends PrimitiveOp implements Iterable<Operand<Integer>> {
  
  /**
   * Factory method to create a class to wrap a new ConcatOffset operation to the graph.
   * 
   * @param scope Current graph scope
   * @param concatDim
   * @param shape
   * @return a new instance of ConcatOffset
   **/
  public static ConcatOffset create(Scope scope, Operand<Integer> concatDim, Iterable<Operand<Integer>> shape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ConcatOffset", scope.makeOpName("ConcatOffset"));
    opBuilder.addInput(concatDim.asOutput());
    opBuilder.addInputList(Operands.asOutputs(shape));
    return new ConcatOffset(opBuilder.build());
  }
  
  public List<Output<Integer>> offset() {
    return offset;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<Integer>> iterator() {
    return (Iterator) offset.iterator();
  }
  
  private List<Output<Integer>> offset;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ConcatOffset(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int offsetLength = operation.outputListLength("offset");
    offset = Arrays.asList((Output<Integer>[])operation.outputList(outputIdx, offsetLength));
    outputIdx += offsetLength;
  }
}
