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

public final class HistogramFixedWidth<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Factory method to create a class to wrap a new HistogramFixedWidth operation to the graph.
   * 
   * @param scope Current graph scope
   * @param values
   * @param valueRange
   * @param nbins
   * @param dtype
   * @return a new instance of HistogramFixedWidth
   **/
  public static <T, U> HistogramFixedWidth<U> create(Scope scope, Operand<T> values, Operand<T> valueRange, Operand<Integer> nbins, Class<U> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("HistogramFixedWidth", scope.makeOpName("HistogramFixedWidth"));
    opBuilder.addInput(values.asOutput());
    opBuilder.addInput(valueRange.asOutput());
    opBuilder.addInput(nbins.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new HistogramFixedWidth<U>(opBuilder.build());
  }
  
  public Output<U> out() {
    return out;
  }
  
  @Override
  public Output<U> asOutput() {
    return out;
  }
  
  private Output<U> out;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private HistogramFixedWidth(Operation operation) {
    super(operation);
    int outputIdx = 0;
    out = operation.output(outputIdx++);
  }
}
