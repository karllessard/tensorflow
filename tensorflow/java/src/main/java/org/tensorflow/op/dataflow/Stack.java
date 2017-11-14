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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Stack extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param stackName
     **/
    public Options stackName(String stackName) {
      this.stackName = stackName;
      return this;
    }
    
    private String stackName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Stack operation to the graph.
   * 
   * @param scope Current graph scope
   * @param elem_type
   * @return a new instance of Stack
   **/
  public static <T> Stack create(Scope scope, Class<T> elem_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Stack", scope.makeOpName("Stack"));
    opBuilder.setAttr("elem_type", DataType.fromClass(elem_type));
    return new Stack(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Stack operation to the graph.
   * 
   * @param scope Current graph scope
   * @param elem_type
   * @param options an object holding optional attributes values
   * @return a new instance of Stack
   **/
  public static <T> Stack create(Scope scope, Class<T> elem_type, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Stack", scope.makeOpName("Stack"));
    opBuilder.setAttr("elem_type", DataType.fromClass(elem_type));
    if (options.stackName != null) {
      opBuilder.setAttr("stackName", options.stackName);
    }
    return new Stack(opBuilder.build());
  }
  
  /**
   * @param stackName
   **/
  public static Options stackName(String stackName) {
    return new Options().stackName(stackName);
  }
  
  public Output<String> handle() {
    return handle;
  }
  
  @Override
  public Output<String> asOutput() {
    return handle;
  }
  
  private Output<String> handle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Stack(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
