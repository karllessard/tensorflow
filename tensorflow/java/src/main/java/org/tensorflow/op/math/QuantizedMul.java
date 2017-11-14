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

public final class QuantizedMul<V> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizedMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param y
   * @param minX
   * @param maxX
   * @param minY
   * @param maxY
   * @param Toutput
   * @return a new instance of QuantizedMul
   **/
  public static <T, U, V> QuantizedMul<V> create(Scope scope, Operand<T> x, Operand<U> y, Operand<Float> minX, Operand<Float> maxX, Operand<Float> minY, Operand<Float> maxY, Class<V> Toutput) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedMul", scope.makeOpName("QuantizedMul"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(y.asOutput());
    opBuilder.addInput(minX.asOutput());
    opBuilder.addInput(maxX.asOutput());
    opBuilder.addInput(minY.asOutput());
    opBuilder.addInput(maxY.asOutput());
    opBuilder.setAttr("Toutput", DataType.fromClass(Toutput));
    return new QuantizedMul<V>(opBuilder.build());
  }
  
  public Output<V> z() {
    return z;
  }
  
  public Output<Float> minZ() {
    return minZ;
  }
  
  public Output<Float> maxZ() {
    return maxZ;
  }
  
  private Output<V> z;
  private Output<Float> minZ;
  private Output<Float> maxZ;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedMul(Operation operation) {
    super(operation);
    int outputIdx = 0;
    z = operation.output(outputIdx++);
    minZ = operation.output(outputIdx++);
    maxZ = operation.output(outputIdx++);
  }
}
