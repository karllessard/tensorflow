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

public final class AsString extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param precision
     **/
    public Options precision(Integer precision) {
      this.precision = precision;
      return this;
    }
    
    /**
     * @param scientific
     **/
    public Options scientific(Boolean scientific) {
      this.scientific = scientific;
      return this;
    }
    
    /**
     * @param shortest
     **/
    public Options shortest(Boolean shortest) {
      this.shortest = shortest;
      return this;
    }
    
    /**
     * @param width
     **/
    public Options width(Integer width) {
      this.width = width;
      return this;
    }
    
    /**
     * @param fill
     **/
    public Options fill(String fill) {
      this.fill = fill;
      return this;
    }
    
    private Integer precision;
    private Boolean scientific;
    private Boolean shortest;
    private Integer width;
    private String fill;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new AsString operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @return a new instance of AsString
   **/
  public static <T> AsString create(Scope scope, Operand<T> input) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AsString", scope.makeOpName("AsString"));
    opBuilder.addInput(input.asOutput());
    return new AsString(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new AsString operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param options an object holding optional attributes values
   * @return a new instance of AsString
   **/
  public static <T> AsString create(Scope scope, Operand<T> input, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AsString", scope.makeOpName("AsString"));
    opBuilder.addInput(input.asOutput());
    if (options.precision != null) {
      opBuilder.setAttr("precision", options.precision);
    }
    if (options.scientific != null) {
      opBuilder.setAttr("scientific", options.scientific);
    }
    if (options.shortest != null) {
      opBuilder.setAttr("shortest", options.shortest);
    }
    if (options.width != null) {
      opBuilder.setAttr("width", options.width);
    }
    if (options.fill != null) {
      opBuilder.setAttr("fill", options.fill);
    }
    return new AsString(opBuilder.build());
  }
  
  /**
   * @param precision
   **/
  public static Options precision(Integer precision) {
    return new Options().precision(precision);
  }
  
  /**
   * @param scientific
   **/
  public static Options scientific(Boolean scientific) {
    return new Options().scientific(scientific);
  }
  
  /**
   * @param shortest
   **/
  public static Options shortest(Boolean shortest) {
    return new Options().shortest(shortest);
  }
  
  /**
   * @param width
   **/
  public static Options width(Integer width) {
    return new Options().width(width);
  }
  
  /**
   * @param fill
   **/
  public static Options fill(String fill) {
    return new Options().fill(fill);
  }
  
  public Output<String> output() {
    return output;
  }
  
  @Override
  public Output<String> asOutput() {
    return output;
  }
  
  private Output<String> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private AsString(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
