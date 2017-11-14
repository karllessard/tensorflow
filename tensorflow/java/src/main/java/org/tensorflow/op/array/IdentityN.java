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

public final class IdentityN extends PrimitiveOp implements Iterable<Operand<Object>> {
  
  /**
   * Factory method to create a class to wrap a new IdentityN operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @return a new instance of IdentityN
   **/
  public static IdentityN create(Scope scope, Iterable<Operand<?>> input) {
    OperationBuilder opBuilder = scope.graph().opBuilder("IdentityN", scope.makeOpName("IdentityN"));
    opBuilder.addInputList(Operands.asOutputs(input));
    return new IdentityN(opBuilder.build());
  }
  
  public List<Output<?>> output() {
    return output;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<Object>> iterator() {
    return (Iterator) output.iterator();
  }
  
  private List<Output<?>> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private IdentityN(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int outputLength = operation.outputListLength("output");
    output = Arrays.asList(operation.outputList(outputIdx, outputLength));
    outputIdx += outputLength;
  }
}
