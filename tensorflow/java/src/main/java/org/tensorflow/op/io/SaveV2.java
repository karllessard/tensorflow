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
package org.tensorflow.op.io;

import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SaveV2 extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new SaveV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param prefix
   * @param tensorNames
   * @param shapeAndSlices
   * @param tensors
   * @return a new instance of SaveV2
   **/
  public static SaveV2 create(Scope scope, Operand<String> prefix, Operand<String> tensorNames, Operand<String> shapeAndSlices, Iterable<Operand<?>> tensors) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SaveV2", scope.makeOpName("SaveV2"));
    opBuilder.addInput(prefix.asOutput());
    opBuilder.addInput(tensorNames.asOutput());
    opBuilder.addInput(shapeAndSlices.asOutput());
    opBuilder.addInputList(Operands.asOutputs(tensors));
    return new SaveV2(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SaveV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
