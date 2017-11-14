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

public final class TensorArrayGradV2 extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new TensorArrayGradV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param flowIn
   * @param source
   * @return a new instance of TensorArrayGradV2
   **/
  public static TensorArrayGradV2 create(Scope scope, Operand<String> handle, Operand<Float> flowIn, String source) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayGradV2", scope.makeOpName("TensorArrayGradV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    opBuilder.setAttr("source", source);
    return new TensorArrayGradV2(opBuilder.build());
  }
  
  public Output<String> gradHandle() {
    return gradHandle;
  }
  
  @Override
  public Output<String> asOutput() {
    return gradHandle;
  }
  
  private Output<String> gradHandle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorArrayGradV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    gradHandle = operation.output(outputIdx++);
  }
}
