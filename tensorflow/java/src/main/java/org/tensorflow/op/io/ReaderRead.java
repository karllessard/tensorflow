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

public final class ReaderRead extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new ReaderRead operation to the graph.
   * 
   * @param scope Current graph scope
   * @param readerHandle
   * @param queueHandle
   * @return a new instance of ReaderRead
   **/
  public static ReaderRead create(Scope scope, Operand<String> readerHandle, Operand<String> queueHandle) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReaderRead", scope.makeOpName("ReaderRead"));
    opBuilder.addInput(readerHandle.asOutput());
    opBuilder.addInput(queueHandle.asOutput());
    return new ReaderRead(opBuilder.build());
  }
  
  public Output<String> key() {
    return key;
  }
  
  public Output<String> value() {
    return value;
  }
  
  private Output<String> key;
  private Output<String> value;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ReaderRead(Operation operation) {
    super(operation);
    int outputIdx = 0;
    key = operation.output(outputIdx++);
    value = operation.output(outputIdx++);
  }
}
