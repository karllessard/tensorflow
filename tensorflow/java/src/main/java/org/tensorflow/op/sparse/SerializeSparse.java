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
package org.tensorflow.op.sparse;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SerializeSparse extends PrimitiveOp implements Operand<String> {
  
  /**
   * Factory method to create a class to wrap a new SerializeSparse operation to the graph.
   * 
   * @param scope Current graph scope
   * @param sparseIndices
   * @param sparseValues
   * @param sparseShape
   * @return a new instance of SerializeSparse
   **/
  public static <T> SerializeSparse create(Scope scope, Operand<Long> sparseIndices, Operand<T> sparseValues, Operand<Long> sparseShape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SerializeSparse", scope.makeOpName("SerializeSparse"));
    opBuilder.addInput(sparseIndices.asOutput());
    opBuilder.addInput(sparseValues.asOutput());
    opBuilder.addInput(sparseShape.asOutput());
    return new SerializeSparse(opBuilder.build());
  }
  
  public Output<String> serializedSparse() {
    return serializedSparse;
  }
  
  @Override
  public Output<String> asOutput() {
    return serializedSparse;
  }
  
  private Output<String> serializedSparse;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SerializeSparse(Operation operation) {
    super(operation);
    int outputIdx = 0;
    serializedSparse = operation.output(outputIdx++);
  }
}
