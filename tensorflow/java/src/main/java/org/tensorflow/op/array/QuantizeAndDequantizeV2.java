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

public final class QuantizeAndDequantizeV2<T> extends PrimitiveOp implements Operand<T> {
  
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
     * @param numBits
     **/
    public Options numBits(Integer numBits) {
      this.numBits = numBits;
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
    private Integer numBits;
    private Boolean rangeGiven;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizeAndDequantizeV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param inputMin
   * @param inputMax
   * @return a new instance of QuantizeAndDequantizeV2
   **/
  public static <T> QuantizeAndDequantizeV2<T> create(Scope scope, Operand<T> input, Operand<T> inputMin, Operand<T> inputMax) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeAndDequantizeV2", scope.makeOpName("QuantizeAndDequantizeV2"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(inputMin.asOutput());
    opBuilder.addInput(inputMax.asOutput());
    return new QuantizeAndDequantizeV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizeAndDequantizeV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param inputMin
   * @param inputMax
   * @param options an object holding optional attributes values
   * @return a new instance of QuantizeAndDequantizeV2
   **/
  public static <T> QuantizeAndDequantizeV2<T> create(Scope scope, Operand<T> input, Operand<T> inputMin, Operand<T> inputMax, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeAndDequantizeV2", scope.makeOpName("QuantizeAndDequantizeV2"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(inputMin.asOutput());
    opBuilder.addInput(inputMax.asOutput());
    if (options.signedInput != null) {
      opBuilder.setAttr("signedInput", options.signedInput);
    }
    if (options.numBits != null) {
      opBuilder.setAttr("numBits", options.numBits);
    }
    if (options.rangeGiven != null) {
      opBuilder.setAttr("rangeGiven", options.rangeGiven);
    }
    return new QuantizeAndDequantizeV2<T>(opBuilder.build());
  }
  
  /**
   * @param signedInput
   **/
  public static Options signedInput(Boolean signedInput) {
    return new Options().signedInput(signedInput);
  }
  
  /**
   * @param numBits
   **/
  public static Options numBits(Integer numBits) {
    return new Options().numBits(numBits);
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
  private QuantizeAndDequantizeV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
