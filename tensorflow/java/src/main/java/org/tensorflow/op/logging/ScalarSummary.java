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

public final class ScalarSummary extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new ScalarSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tags
   * @param values
   * @return a new instance of ScalarSummary
   **/
  public static <T> ScalarSummary create(Scope scope, Operand<String> tags, Operand<T> values) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ScalarSummary", scope.makeOpName("ScalarSummary"));
    opBuilder.addInput(tags.asOutput());
    opBuilder.addInput(values.asOutput());
    return new ScalarSummary(opBuilder.build());
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
  private ScalarSummary(Operation operation) {
    super(operation);
    int outputIdx = 0;
    summary = operation.output(outputIdx++);
  }
}
