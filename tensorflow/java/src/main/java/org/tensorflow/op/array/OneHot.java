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
package org.tensorflow.op.array;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class OneHot<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param axis
     **/
    public Options axis(Integer axis) {
      this.axis = axis;
      return this;
    }
    
    private Integer axis;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new OneHot operation to the graph.
   * 
   * @param scope Current graph scope
   * @param indices
   * @param depth
   * @param onValue
   * @param offValue
   * @return a new instance of OneHot
   **/
  public static <T, U> OneHot<U> create(Scope scope, Operand<T> indices, Operand<Integer> depth, Operand<U> onValue, Operand<U> offValue) {
    OperationBuilder opBuilder = scope.graph().opBuilder("OneHot", scope.makeOpName("OneHot"));
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(depth.asOutput());
    opBuilder.addInput(onValue.asOutput());
    opBuilder.addInput(offValue.asOutput());
    return new OneHot<U>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new OneHot operation to the graph.
   * 
   * @param scope Current graph scope
   * @param indices
   * @param depth
   * @param onValue
   * @param offValue
   * @param options an object holding optional attributes values
   * @return a new instance of OneHot
   **/
  public static <T, U> OneHot<U> create(Scope scope, Operand<T> indices, Operand<Integer> depth, Operand<U> onValue, Operand<U> offValue, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("OneHot", scope.makeOpName("OneHot"));
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(depth.asOutput());
    opBuilder.addInput(onValue.asOutput());
    opBuilder.addInput(offValue.asOutput());
    if (options.axis != null) {
      opBuilder.setAttr("axis", options.axis);
    }
    return new OneHot<U>(opBuilder.build());
  }
  
  /**
   * @param axis
   **/
  public static Options axis(Integer axis) {
    return new Options().axis(axis);
  }
  
  public Output<U> output() {
    return output;
  }
  
  @Override
  public Output<U> asOutput() {
    return output;
  }
  
  private Output<U> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private OneHot(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
