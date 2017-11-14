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

public final class UniqueV2<T, U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new UniqueV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param axis
   * @param out_idx
   * @return a new instance of UniqueV2
   **/
  public static <T, U> UniqueV2<T, U> create(Scope scope, Operand<T> x, Operand<Long> axis, Class<U> out_idx) {
    OperationBuilder opBuilder = scope.graph().opBuilder("UniqueV2", scope.makeOpName("UniqueV2"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(axis.asOutput());
    opBuilder.setAttr("out_idx", DataType.fromClass(out_idx));
    return new UniqueV2<T, U>(opBuilder.build());
  }
  
  public Output<T> y() {
    return y;
  }
  
  public Output<U> idx() {
    return idx;
  }
  
  private Output<T> y;
  private Output<U> idx;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private UniqueV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    y = operation.output(outputIdx++);
    idx = operation.output(outputIdx++);
  }
}
