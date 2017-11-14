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

public final class TemporaryVariable<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param varName
     **/
    public Options varName(String varName) {
      this.varName = varName;
      return this;
    }
    
    private String varName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TemporaryVariable operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param dtype
   * @return a new instance of TemporaryVariable
   **/
  public static <T> TemporaryVariable<T> create(Scope scope, Shape shape, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TemporaryVariable", scope.makeOpName("TemporaryVariable"));
    opBuilder.setAttr("shape", shape);
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new TemporaryVariable<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TemporaryVariable operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of TemporaryVariable
   **/
  public static <T> TemporaryVariable<T> create(Scope scope, Shape shape, Class<T> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TemporaryVariable", scope.makeOpName("TemporaryVariable"));
    opBuilder.setAttr("shape", shape);
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.varName != null) {
      opBuilder.setAttr("varName", options.varName);
    }
    return new TemporaryVariable<T>(opBuilder.build());
  }
  
  /**
   * @param varName
   **/
  public static Options varName(String varName) {
    return new Options().varName(varName);
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
  private TemporaryVariable(Operation operation) {
    super(operation);
    int outputIdx = 0;
    ref = operation.output(outputIdx++);
  }
}
