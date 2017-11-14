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
package org.tensorflow.op.dataflow;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class TensorArraySplitV3 extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Factory method to create a class to wrap a new TensorArraySplitV3 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param value
   * @param lengths
   * @param flowIn
   * @return a new instance of TensorArraySplitV3
   **/
  public static <T> TensorArraySplitV3 create(Scope scope, Operand<?> handle, Operand<T> value, Operand<Long> lengths, Operand<Float> flowIn) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArraySplitV3", scope.makeOpName("TensorArraySplitV3"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(value.asOutput());
    opBuilder.addInput(lengths.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    return new TensorArraySplitV3(opBuilder.build());
  }
  
  public Output<Float> flowOut() {
    return flowOut;
  }
  
  @Override
  public Output<Float> asOutput() {
    return flowOut;
  }
  
  private Output<Float> flowOut;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorArraySplitV3(Operation operation) {
    super(operation);
    int outputIdx = 0;
    flowOut = operation.output(outputIdx++);
  }
}
