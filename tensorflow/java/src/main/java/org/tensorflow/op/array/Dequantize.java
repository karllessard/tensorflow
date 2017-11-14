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

public final class Dequantize extends PrimitiveOp implements Operand<Float> {
  
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
    
    private String mode;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Dequantize operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param minRange
   * @param maxRange
   * @return a new instance of Dequantize
   **/
  public static <T> Dequantize create(Scope scope, Operand<T> input, Operand<Float> minRange, Operand<Float> maxRange) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Dequantize", scope.makeOpName("Dequantize"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(minRange.asOutput());
    opBuilder.addInput(maxRange.asOutput());
    return new Dequantize(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Dequantize operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param minRange
   * @param maxRange
   * @param options an object holding optional attributes values
   * @return a new instance of Dequantize
   **/
  public static <T> Dequantize create(Scope scope, Operand<T> input, Operand<Float> minRange, Operand<Float> maxRange, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Dequantize", scope.makeOpName("Dequantize"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(minRange.asOutput());
    opBuilder.addInput(maxRange.asOutput());
    if (options.mode != null) {
      opBuilder.setAttr("mode", options.mode);
    }
    return new Dequantize(opBuilder.build());
  }
  
  /**
   * @param mode
   **/
  public static Options mode(String mode) {
    return new Options().mode(mode);
  }
  
  public Output<Float> output() {
    return output;
  }
  
  @Override
  public Output<Float> asOutput() {
    return output;
  }
  
  private Output<Float> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Dequantize(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
