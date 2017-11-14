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
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ReaderRestoreStateV2 extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new ReaderRestoreStateV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param readerHandle
   * @param state
   * @return a new instance of ReaderRestoreStateV2
   **/
  public static ReaderRestoreStateV2 create(Scope scope, Operand<?> readerHandle, Operand<String> state) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReaderRestoreStateV2", scope.makeOpName("ReaderRestoreStateV2"));
    opBuilder.addInput(readerHandle.asOutput());
    opBuilder.addInput(state.asOutput());
    return new ReaderRestoreStateV2(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ReaderRestoreStateV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
