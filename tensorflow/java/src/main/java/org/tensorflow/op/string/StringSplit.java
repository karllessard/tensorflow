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
package org.tensorflow.op.string;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class StringSplit extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param skipEmpty
     **/
    public Options skipEmpty(Boolean skipEmpty) {
      this.skipEmpty = skipEmpty;
      return this;
    }
    
    private Boolean skipEmpty;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new StringSplit operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param delimiter
   * @return a new instance of StringSplit
   **/
  public static StringSplit create(Scope scope, Operand<String> input, Operand<String> delimiter) {
    OperationBuilder opBuilder = scope.graph().opBuilder("StringSplit", scope.makeOpName("StringSplit"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(delimiter.asOutput());
    return new StringSplit(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new StringSplit operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param delimiter
   * @param options an object holding optional attributes values
   * @return a new instance of StringSplit
   **/
  public static StringSplit create(Scope scope, Operand<String> input, Operand<String> delimiter, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("StringSplit", scope.makeOpName("StringSplit"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(delimiter.asOutput());
    if (options.skipEmpty != null) {
      opBuilder.setAttr("skipEmpty", options.skipEmpty);
    }
    return new StringSplit(opBuilder.build());
  }
  
  /**
   * @param skipEmpty
   **/
  public static Options skipEmpty(Boolean skipEmpty) {
    return new Options().skipEmpty(skipEmpty);
  }
  
  public Output<Long> indices() {
    return indices;
  }
  
  public Output<String> values() {
    return values;
  }
  
  public Output<Long> shape() {
    return shape;
  }
  
  private Output<Long> indices;
  private Output<String> values;
  private Output<Long> shape;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private StringSplit(Operation operation) {
    super(operation);
    int outputIdx = 0;
    indices = operation.output(outputIdx++);
    values = operation.output(outputIdx++);
    shape = operation.output(outputIdx++);
  }
}
