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
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QueueEnqueue extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param timeoutMs
     **/
    public Options timeoutMs(Integer timeoutMs) {
      this.timeoutMs = timeoutMs;
      return this;
    }
    
    private Integer timeoutMs;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QueueEnqueue operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param components
   * @return a new instance of QueueEnqueue
   **/
  public static QueueEnqueue create(Scope scope, Operand<String> handle, Iterable<Operand<?>> components) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QueueEnqueue", scope.makeOpName("QueueEnqueue"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInputList(Operands.asOutputs(components));
    return new QueueEnqueue(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QueueEnqueue operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param components
   * @param options an object holding optional attributes values
   * @return a new instance of QueueEnqueue
   **/
  public static QueueEnqueue create(Scope scope, Operand<String> handle, Iterable<Operand<?>> components, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QueueEnqueue", scope.makeOpName("QueueEnqueue"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInputList(Operands.asOutputs(components));
    if (options.timeoutMs != null) {
      opBuilder.setAttr("timeoutMs", options.timeoutMs);
    }
    return new QueueEnqueue(opBuilder.build());
  }
  
  /**
   * @param timeoutMs
   **/
  public static Options timeoutMs(Integer timeoutMs) {
    return new Options().timeoutMs(timeoutMs);
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QueueEnqueue(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
