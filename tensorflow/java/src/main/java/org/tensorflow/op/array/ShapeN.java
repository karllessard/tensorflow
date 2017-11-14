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
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ShapeN<U> extends PrimitiveOp implements Iterable<Operand<U>> {
  
  /**
   * Factory method to create a class to wrap a new ShapeN operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param out_type
   * @return a new instance of ShapeN
   **/
  public static <T, U> ShapeN<U> create(Scope scope, Iterable<Operand<T>> input, Class<U> out_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ShapeN", scope.makeOpName("ShapeN"));
    opBuilder.addInputList(Operands.asOutputs(input));
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    return new ShapeN<U>(opBuilder.build());
  }
  
  public List<Output<U>> output() {
    return output;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<U>> iterator() {
    return (Iterator) output.iterator();
  }
  
  private List<Output<U>> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ShapeN(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int outputLength = operation.outputListLength("output");
    output = Arrays.asList((Output<U>[])operation.outputList(outputIdx, outputLength));
    outputIdx += outputLength;
  }
}
