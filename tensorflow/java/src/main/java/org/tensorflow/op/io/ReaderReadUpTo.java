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

public final class ReaderReadUpTo extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new ReaderReadUpTo operation to the graph.
   * 
   * @param scope Current graph scope
   * @param readerHandle
   * @param queueHandle
   * @param numRecords
   * @return a new instance of ReaderReadUpTo
   **/
  public static ReaderReadUpTo create(Scope scope, Operand<String> readerHandle, Operand<String> queueHandle, Operand<Long> numRecords) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReaderReadUpTo", scope.makeOpName("ReaderReadUpTo"));
    opBuilder.addInput(readerHandle.asOutput());
    opBuilder.addInput(queueHandle.asOutput());
    opBuilder.addInput(numRecords.asOutput());
    return new ReaderReadUpTo(opBuilder.build());
  }
  
  public Output<String> keys() {
    return keys;
  }
  
  public Output<String> values() {
    return values;
  }
  
  private Output<String> keys;
  private Output<String> values;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ReaderReadUpTo(Operation operation) {
    super(operation);
    int outputIdx = 0;
    keys = operation.output(outputIdx++);
    values = operation.output(outputIdx++);
  }
}
