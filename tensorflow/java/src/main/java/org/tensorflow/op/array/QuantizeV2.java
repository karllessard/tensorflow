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
import org.tensorflow.op.Scope;

public final class QuantizeV2<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param mode
     **/
    public Options mode(String mode) {
      this.mode = mode;
      return this;
    }
    
    /**
     * @param roundMode
     **/
    public Options roundMode(String roundMode) {
      this.roundMode = roundMode;
      return this;
    }
    
    private String mode;
    private String roundMode;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizeV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param minRange
   * @param maxRange
   * @param T
   * @return a new instance of QuantizeV2
   **/
  public static <T> QuantizeV2<T> create(Scope scope, Operand<Float> input, Operand<Float> minRange, Operand<Float> maxRange, Class<T> T) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeV2", scope.makeOpName("QuantizeV2"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(minRange.asOutput());
    opBuilder.addInput(maxRange.asOutput());
    opBuilder.setAttr("T", DataType.fromClass(T));
    return new QuantizeV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizeV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param minRange
   * @param maxRange
   * @param T
   * @param options an object holding optional attributes values
   * @return a new instance of QuantizeV2
   **/
  public static <T> QuantizeV2<T> create(Scope scope, Operand<Float> input, Operand<Float> minRange, Operand<Float> maxRange, Class<T> T, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeV2", scope.makeOpName("QuantizeV2"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(minRange.asOutput());
    opBuilder.addInput(maxRange.asOutput());
    opBuilder.setAttr("T", DataType.fromClass(T));
    if (options.mode != null) {
      opBuilder.setAttr("mode", options.mode);
    }
    if (options.roundMode != null) {
      opBuilder.setAttr("roundMode", options.roundMode);
    }
    return new QuantizeV2<T>(opBuilder.build());
  }
  
  /**
   * @param mode
   **/
  public static Options mode(String mode) {
    return new Options().mode(mode);
  }
  
  /**
   * @param roundMode
   **/
  public static Options roundMode(String roundMode) {
    return new Options().roundMode(roundMode);
  }
  
  public Output<T> output() {
    return output;
  }
  
  public Output<Float> outputMin() {
    return outputMin;
  }
  
  public Output<Float> outputMax() {
    return outputMax;
  }
  
  private Output<T> output;
  private Output<Float> outputMin;
  private Output<Float> outputMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizeV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    outputMin = operation.output(outputIdx++);
    outputMax = operation.output(outputIdx++);
  }
}
