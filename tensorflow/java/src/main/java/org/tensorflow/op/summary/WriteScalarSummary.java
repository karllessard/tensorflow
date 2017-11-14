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

public final class WriteScalarSummary extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new WriteScalarSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param writer
   * @param globalStep
   * @param tag
   * @param value
   * @return a new instance of WriteScalarSummary
   **/
  public static <T> WriteScalarSummary create(Scope scope, Operand<?> writer, Operand<Long> globalStep, Operand<String> tag, Operand<T> value) {
    OperationBuilder opBuilder = scope.graph().opBuilder("WriteScalarSummary", scope.makeOpName("WriteScalarSummary"));
    opBuilder.addInput(writer.asOutput());
    opBuilder.addInput(globalStep.asOutput());
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(value.asOutput());
    return new WriteScalarSummary(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private WriteScalarSummary(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
