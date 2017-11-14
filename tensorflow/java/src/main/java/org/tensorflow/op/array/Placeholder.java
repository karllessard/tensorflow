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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class Placeholder<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param shape
     **/
    public Options shape(Shape shape) {
      this.shape = shape;
      return this;
    }
    
    private Shape shape;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Placeholder operation to the graph.
   * 
   * @param scope Current graph scope
   * @param dtype
   * @return a new instance of Placeholder
   **/
  public static <T> Placeholder<T> create(Scope scope, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Placeholder", scope.makeOpName("Placeholder"));
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new Placeholder<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Placeholder operation to the graph.
   * 
   * @param scope Current graph scope
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of Placeholder
   **/
  public static <T> Placeholder<T> create(Scope scope, Class<T> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Placeholder", scope.makeOpName("Placeholder"));
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.shape != null) {
      opBuilder.setAttr("shape", options.shape);
    }
    return new Placeholder<T>(opBuilder.build());
  }
  
  /**
   * @param shape
   **/
  public static Options shape(Shape shape) {
    return new Options().shape(shape);
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
  private Placeholder(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
