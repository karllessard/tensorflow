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
package org.tensorflow.op.logging;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class TensorSummaryV2 extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new TensorSummaryV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tag
   * @param tensor
   * @param serializedSummaryMetadata
   * @return a new instance of TensorSummaryV2
   **/
  public static <T> TensorSummaryV2 create(Scope scope, Operand<String> tag, Operand<T> tensor, Operand<String> serializedSummaryMetadata) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorSummaryV2", scope.makeOpName("TensorSummaryV2"));
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(tensor.asOutput());
    opBuilder.addInput(serializedSummaryMetadata.asOutput());
    return new TensorSummaryV2(opBuilder.build());
  }
  
  public Output<String> summary() {
    return summary;
  }
  
  @Override
  public Output<String> asOutput() {
    return summary;
  }
  
  private Output<String> summary;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorSummaryV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    summary = operation.output(outputIdx++);
  }
}
