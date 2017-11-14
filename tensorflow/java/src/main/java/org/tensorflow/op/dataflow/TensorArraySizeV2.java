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

public final class TensorArraySizeV2 extends PrimitiveOp implements Operand<Integer> {
  
  /**
   * Factory method to create a class to wrap a new TensorArraySizeV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param flowIn
   * @return a new instance of TensorArraySizeV2
   **/
  public static TensorArraySizeV2 create(Scope scope, Operand<String> handle, Operand<Float> flowIn) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArraySizeV2", scope.makeOpName("TensorArraySizeV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    return new TensorArraySizeV2(opBuilder.build());
  }
  
  public Output<Integer> size() {
    return size;
  }
  
  @Override
  public Output<Integer> asOutput() {
    return size;
  }
  
  private Output<Integer> size;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorArraySizeV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    size = operation.output(outputIdx++);
  }
}
