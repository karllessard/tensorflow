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

public final class UniqueWithCounts<T, U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new UniqueWithCounts operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param out_idx
   * @return a new instance of UniqueWithCounts
   **/
  public static <T, U> UniqueWithCounts<T, U> create(Scope scope, Operand<T> x, Class<U> out_idx) {
    OperationBuilder opBuilder = scope.graph().opBuilder("UniqueWithCounts", scope.makeOpName("UniqueWithCounts"));
    opBuilder.addInput(x.asOutput());
    opBuilder.setAttr("out_idx", DataType.fromClass(out_idx));
    return new UniqueWithCounts<T, U>(opBuilder.build());
  }
  
  public Output<T> y() {
    return y;
  }
  
  public Output<U> idx() {
    return idx;
  }
  
  public Output<U> count() {
    return count;
  }
  
  private Output<T> y;
  private Output<U> idx;
  private Output<U> count;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private UniqueWithCounts(Operation operation) {
    super(operation);
    int outputIdx = 0;
    y = operation.output(outputIdx++);
    idx = operation.output(outputIdx++);
    count = operation.output(outputIdx++);
  }
}
