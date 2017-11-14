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
package org.tensorflow.op.logging;

import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Print<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param message
     **/
    public Options message(String message) {
      this.message = message;
      return this;
    }
    
    /**
     * @param firstN
     **/
    public Options firstN(Integer firstN) {
      this.firstN = firstN;
      return this;
    }
    
    /**
     * @param summarize
     **/
    public Options summarize(Integer summarize) {
      this.summarize = summarize;
      return this;
    }
    
    private String message;
    private Integer firstN;
    private Integer summarize;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Print operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param data
   * @return a new instance of Print
   **/
  public static <T> Print<T> create(Scope scope, Operand<T> input, Iterable<Operand<?>> data) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Print", scope.makeOpName("Print"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInputList(Operands.asOutputs(data));
    return new Print<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Print operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param data
   * @param options an object holding optional attributes values
   * @return a new instance of Print
   **/
  public static <T> Print<T> create(Scope scope, Operand<T> input, Iterable<Operand<?>> data, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Print", scope.makeOpName("Print"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInputList(Operands.asOutputs(data));
    if (options.message != null) {
      opBuilder.setAttr("message", options.message);
    }
    if (options.firstN != null) {
      opBuilder.setAttr("firstN", options.firstN);
    }
    if (options.summarize != null) {
      opBuilder.setAttr("summarize", options.summarize);
    }
    return new Print<T>(opBuilder.build());
  }
  
  /**
   * @param message
   **/
  public static Options message(String message) {
    return new Options().message(message);
  }
  
  /**
   * @param firstN
   **/
  public static Options firstN(Integer firstN) {
    return new Options().firstN(firstN);
  }
  
  /**
   * @param summarize
   **/
  public static Options summarize(Integer summarize) {
    return new Options().summarize(summarize);
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
  private Print(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
