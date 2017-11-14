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
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class BarrierInsertMany extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new BarrierInsertMany operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param keys
   * @param values
   * @param componentIndex
   * @return a new instance of BarrierInsertMany
   **/
  public static <T> BarrierInsertMany create(Scope scope, Operand<String> handle, Operand<String> keys, Operand<T> values, Integer componentIndex) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BarrierInsertMany", scope.makeOpName("BarrierInsertMany"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(keys.asOutput());
    opBuilder.addInput(values.asOutput());
    opBuilder.setAttr("componentIndex", componentIndex);
    return new BarrierInsertMany(opBuilder.build());
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private BarrierInsertMany(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
