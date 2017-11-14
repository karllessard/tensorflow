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

public final class QuantizeAndDequantizeV3<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param signedInput
     **/
    public Options signedInput(Boolean signedInput) {
      this.signedInput = signedInput;
      return this;
    }
    
    /**
     * @param rangeGiven
     **/
    public Options rangeGiven(Boolean rangeGiven) {
      this.rangeGiven = rangeGiven;
      return this;
    }
    
    private Boolean signedInput;
    private Boolean rangeGiven;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizeAndDequantizeV3 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param inputMin
   * @param inputMax
   * @param numBits
   * @return a new instance of QuantizeAndDequantizeV3
   **/
  public static <T> QuantizeAndDequantizeV3<T> create(Scope scope, Operand<T> input, Operand<T> inputMin, Operand<T> inputMax, Operand<Integer> numBits) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeAndDequantizeV3", scope.makeOpName("QuantizeAndDequantizeV3"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(inputMin.asOutput());
    opBuilder.addInput(inputMax.asOutput());
    opBuilder.addInput(numBits.asOutput());
    return new QuantizeAndDequantizeV3<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizeAndDequantizeV3 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param inputMin
   * @param inputMax
   * @param numBits
   * @param options an object holding optional attributes values
   * @return a new instance of QuantizeAndDequantizeV3
   **/
  public static <T> QuantizeAndDequantizeV3<T> create(Scope scope, Operand<T> input, Operand<T> inputMin, Operand<T> inputMax, Operand<Integer> numBits, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeAndDequantizeV3", scope.makeOpName("QuantizeAndDequantizeV3"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(inputMin.asOutput());
    opBuilder.addInput(inputMax.asOutput());
    opBuilder.addInput(numBits.asOutput());
    if (options.signedInput != null) {
      opBuilder.setAttr("signedInput", options.signedInput);
    }
    if (options.rangeGiven != null) {
      opBuilder.setAttr("rangeGiven", options.rangeGiven);
    }
    return new QuantizeAndDequantizeV3<T>(opBuilder.build());
  }
  
  /**
   * @param signedInput
   **/
  public static Options signedInput(Boolean signedInput) {
    return new Options().signedInput(signedInput);
  }
  
  /**
   * @param rangeGiven
   **/
  public static Options rangeGiven(Boolean rangeGiven) {
    return new Options().rangeGiven(rangeGiven);
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
  private QuantizeAndDequantizeV3(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
