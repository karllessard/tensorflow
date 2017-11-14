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
package org.tensorflow.op.summary;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class WriteHistogramSummary extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new WriteHistogramSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param writer
   * @param globalStep
   * @param tag
   * @param values
   * @return a new instance of WriteHistogramSummary
   **/
  public static <T> WriteHistogramSummary create(Scope scope, Operand<?> writer, Operand<Long> globalStep, Operand<String> tag, Operand<T> values) {
    OperationBuilder opBuilder = scope.graph().opBuilder("WriteHistogramSummary", scope.makeOpName("WriteHistogramSummary"));
    opBuilder.addInput(writer.asOutput());
    opBuilder.addInput(globalStep.asOutput());
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(values.asOutput());
    return new WriteHistogramSummary(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private WriteHistogramSummary(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
