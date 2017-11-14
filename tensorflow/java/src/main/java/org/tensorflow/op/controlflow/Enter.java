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
package org.tensorflow.op.controlflow;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Enter<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param isConstant
     **/
    public Options isConstant(Boolean isConstant) {
      this.isConstant = isConstant;
      return this;
    }
    
    /**
     * @param parallelIterations
     **/
    public Options parallelIterations(Integer parallelIterations) {
      this.parallelIterations = parallelIterations;
      return this;
    }
    
    private Boolean isConstant;
    private Integer parallelIterations;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Enter operation to the graph.
   * 
   * @param scope Current graph scope
   * @param data
   * @param frameName
   * @return a new instance of Enter
   **/
  public static <T> Enter<T> create(Scope scope, Operand<T> data, String frameName) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Enter", scope.makeOpName("Enter"));
    opBuilder.addInput(data.asOutput());
    opBuilder.setAttr("frameName", frameName);
    return new Enter<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Enter operation to the graph.
   * 
   * @param scope Current graph scope
   * @param data
   * @param frameName
   * @param options an object holding optional attributes values
   * @return a new instance of Enter
   **/
  public static <T> Enter<T> create(Scope scope, Operand<T> data, String frameName, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Enter", scope.makeOpName("Enter"));
    opBuilder.addInput(data.asOutput());
    opBuilder.setAttr("frameName", frameName);
    if (options.isConstant != null) {
      opBuilder.setAttr("isConstant", options.isConstant);
    }
    if (options.parallelIterations != null) {
      opBuilder.setAttr("parallelIterations", options.parallelIterations);
    }
    return new Enter<T>(opBuilder.build());
  }
  
  /**
   * @param isConstant
   **/
  public static Options isConstant(Boolean isConstant) {
    return new Options().isConstant(isConstant);
  }
  
  /**
   * @param parallelIterations
   **/
  public static Options parallelIterations(Integer parallelIterations) {
    return new Options().parallelIterations(parallelIterations);
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
  private Enter(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
