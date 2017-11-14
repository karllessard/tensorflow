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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class BroadcastArgs<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new BroadcastArgs operation to the graph.
   * 
   * @param scope Current graph scope
   * @param s0
   * @param s1
   * @return a new instance of BroadcastArgs
   **/
  public static <T> BroadcastArgs<T> create(Scope scope, Operand<T> s0, Operand<T> s1) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BroadcastArgs", scope.makeOpName("BroadcastArgs"));
    opBuilder.addInput(s0.asOutput());
    opBuilder.addInput(s1.asOutput());
    return new BroadcastArgs<T>(opBuilder.build());
  }
  
  public Output<T> r0() {
    return r0;
  }
  
  @Override
  public Output<T> asOutput() {
    return r0;
  }
  
  private Output<T> r0;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private BroadcastArgs(Operation operation) {
    super(operation);
    int outputIdx = 0;
    r0 = operation.output(outputIdx++);
  }
}
