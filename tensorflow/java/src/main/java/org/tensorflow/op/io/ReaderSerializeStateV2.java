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

public final class ReaderSerializeStateV2 extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new ReaderSerializeStateV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param readerHandle
   * @return a new instance of ReaderSerializeStateV2
   **/
  public static ReaderSerializeStateV2 create(Scope scope, Operand<?> readerHandle) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReaderSerializeStateV2", scope.makeOpName("ReaderSerializeStateV2"));
    opBuilder.addInput(readerHandle.asOutput());
    return new ReaderSerializeStateV2(opBuilder.build());
  }
  
  public Output<String> state() {
    return state;
  }
  
  @Override
  public Output<String> asOutput() {
    return state;
  }
  
  private Output<String> state;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ReaderSerializeStateV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    state = operation.output(outputIdx++);
  }
}
