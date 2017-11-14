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
package org.tensorflow.op.math;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Cast<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Factory method to create a class to wrap a new Cast operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param DstT
   * @return a new instance of Cast
   **/
  public static <T, U> Cast<U> create(Scope scope, Operand<T> x, Class<U> DstT) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Cast", scope.makeOpName("Cast"));
    opBuilder.addInput(x.asOutput());
    opBuilder.setAttr("DstT", DataType.fromClass(DstT));
    return new Cast<U>(opBuilder.build());
  }
  
  public Output<U> y() {
    return y;
  }
  
  @Override
  public Output<U> asOutput() {
    return y;
  }
  
  private Output<U> y;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Cast(Operation operation) {
    super(operation);
    int outputIdx = 0;
    y = operation.output(outputIdx++);
  }
}
