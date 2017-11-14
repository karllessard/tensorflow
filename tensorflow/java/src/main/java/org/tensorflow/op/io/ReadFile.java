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
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ReadFile extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new ReadFile operation to the graph.
   * 
   * @param scope Current graph scope
   * @param filename
   * @return a new instance of ReadFile
   **/
  public static ReadFile create(Scope scope, Operand<String> filename) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReadFile", scope.makeOpName("ReadFile"));
    opBuilder.addInput(filename.asOutput());
    return new ReadFile(opBuilder.build());
  }
  
  public Output<String> contents() {
    return contents;
  }
  
  @Override
  public Output<String> asOutput() {
    return contents;
  }
  
  private Output<String> contents;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ReadFile(Operation operation) {
    super(operation);
    int outputIdx = 0;
    contents = operation.output(outputIdx++);
  }
}
