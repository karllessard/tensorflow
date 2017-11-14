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
package org.tensorflow.op.state;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class Variable<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param container
     **/
    public Options container(String container) {
      this.container = container;
      return this;
    }
    
    /**
     * @param sharedName
     **/
    public Options sharedName(String sharedName) {
      this.sharedName = sharedName;
      return this;
    }
    
    private String container;
    private String sharedName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Variable operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param dtype
   * @return a new instance of Variable
   **/
  public static <T> Variable<T> create(Scope scope, Shape shape, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Variable", scope.makeOpName("Variable"));
    opBuilder.setAttr("shape", shape);
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new Variable<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Variable operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of Variable
   **/
  public static <T> Variable<T> create(Scope scope, Shape shape, Class<T> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Variable", scope.makeOpName("Variable"));
    opBuilder.setAttr("shape", shape);
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    return new Variable<T>(opBuilder.build());
  }
  
  /**
   * @param container
   **/
  public static Options container(String container) {
    return new Options().container(container);
  }
  
  /**
   * @param sharedName
   **/
  public static Options sharedName(String sharedName) {
    return new Options().sharedName(sharedName);
  }
  
  public Output<T> ref() {
    return ref;
  }
  
  @Override
  public Output<T> asOutput() {
    return ref;
  }
  
  private Output<T> ref;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Variable(Operation operation) {
    super(operation);
    int outputIdx = 0;
    ref = operation.output(outputIdx++);
  }
}
