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

public final class StackPush<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param swapMemory
     **/
    public Options swapMemory(Boolean swapMemory) {
      this.swapMemory = swapMemory;
      return this;
    }
    
    private Boolean swapMemory;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new StackPush operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param elem
   * @return a new instance of StackPush
   **/
  public static <T> StackPush<T> create(Scope scope, Operand<String> handle, Operand<T> elem) {
    OperationBuilder opBuilder = scope.graph().opBuilder("StackPush", scope.makeOpName("StackPush"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(elem.asOutput());
    return new StackPush<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new StackPush operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param elem
   * @param options an object holding optional attributes values
   * @return a new instance of StackPush
   **/
  public static <T> StackPush<T> create(Scope scope, Operand<String> handle, Operand<T> elem, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("StackPush", scope.makeOpName("StackPush"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(elem.asOutput());
    if (options.swapMemory != null) {
      opBuilder.setAttr("swapMemory", options.swapMemory);
    }
    return new StackPush<T>(opBuilder.build());
  }
  
  /**
   * @param swapMemory
   **/
  public static Options swapMemory(Boolean swapMemory) {
    return new Options().swapMemory(swapMemory);
  }
  
  public Output<T> output() {
    return output;
  }
  
  @Override
  public Output<T> asOutput() {
    return output;
  }
  
  private Output<T> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private StackPush(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
