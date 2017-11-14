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
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class ImmutableConst<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new ImmutableConst operation to the graph.
   * 
   * @param scope Current graph scope
   * @param dtype
   * @param shape
   * @param memoryRegionName
   * @return a new instance of ImmutableConst
   **/
  public static <T> ImmutableConst<T> create(Scope scope, Class<T> dtype, Shape shape, String memoryRegionName) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ImmutableConst", scope.makeOpName("ImmutableConst"));
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    opBuilder.setAttr("shape", shape);
    opBuilder.setAttr("memoryRegionName", memoryRegionName);
    return new ImmutableConst<T>(opBuilder.build());
  }
  
  public Output<T> tensor() {
    return tensor;
  }
  
  @Override
  public Output<T> asOutput() {
    return tensor;
  }
  
  private Output<T> tensor;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ImmutableConst(Operation operation) {
    super(operation);
    int outputIdx = 0;
    tensor = operation.output(outputIdx++);
  }
}
