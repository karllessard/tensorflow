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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ListDiff<T, U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new ListDiff operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param y
   * @param out_idx
   * @return a new instance of ListDiff
   **/
  public static <T, U> ListDiff<T, U> create(Scope scope, Operand<T> x, Operand<T> y, Class<U> out_idx) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ListDiff", scope.makeOpName("ListDiff"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(y.asOutput());
    opBuilder.setAttr("out_idx", DataType.fromClass(out_idx));
    return new ListDiff<T, U>(opBuilder.build());
  }
  
  public Output<T> out() {
    return out;
  }
  
  public Output<U> idx() {
    return idx;
  }
  
  private Output<T> out;
  private Output<U> idx;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ListDiff(Operation operation) {
    super(operation);
    int outputIdx = 0;
    out = operation.output(outputIdx++);
    idx = operation.output(outputIdx++);
  }
}
