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
package org.tensorflow.op.math;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Cumsum<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param exclusive
     **/
    public Options exclusive(Boolean exclusive) {
      this.exclusive = exclusive;
      return this;
    }
    
    /**
     * @param reverse
     **/
    public Options reverse(Boolean reverse) {
      this.reverse = reverse;
      return this;
    }
    
    private Boolean exclusive;
    private Boolean reverse;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Cumsum operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param axis
   * @return a new instance of Cumsum
   **/
  public static <T, U> Cumsum<T> create(Scope scope, Operand<T> x, Operand<U> axis) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Cumsum", scope.makeOpName("Cumsum"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(axis.asOutput());
    return new Cumsum<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Cumsum operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param axis
   * @param options an object holding optional attributes values
   * @return a new instance of Cumsum
   **/
  public static <T, U> Cumsum<T> create(Scope scope, Operand<T> x, Operand<U> axis, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Cumsum", scope.makeOpName("Cumsum"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(axis.asOutput());
    if (options.exclusive != null) {
      opBuilder.setAttr("exclusive", options.exclusive);
    }
    if (options.reverse != null) {
      opBuilder.setAttr("reverse", options.reverse);
    }
    return new Cumsum<T>(opBuilder.build());
  }
  
  /**
   * @param exclusive
   **/
  public static Options exclusive(Boolean exclusive) {
    return new Options().exclusive(exclusive);
  }
  
  /**
   * @param reverse
   **/
  public static Options reverse(Boolean reverse) {
    return new Options().reverse(reverse);
  }
  
  public Output<T> out() {
    return out;
  }
  
  @Override
  public Output<T> asOutput() {
    return out;
  }
  
  private Output<T> out;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Cumsum(Operation operation) {
    super(operation);
    int outputIdx = 0;
    out = operation.output(outputIdx++);
  }
}
