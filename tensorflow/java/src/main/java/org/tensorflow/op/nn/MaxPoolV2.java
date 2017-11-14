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
package org.tensorflow.op.nn;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class MaxPoolV2<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param dataFormat
     **/
    public Options dataFormat(String dataFormat) {
      this.dataFormat = dataFormat;
      return this;
    }
    
    private String dataFormat;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new MaxPoolV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param ksize
   * @param strides
   * @param padding
   * @return a new instance of MaxPoolV2
   **/
  public static <T> MaxPoolV2<T> create(Scope scope, Operand<T> input, Operand<Integer> ksize, Operand<Integer> strides, String padding) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MaxPoolV2", scope.makeOpName("MaxPoolV2"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(ksize.asOutput());
    opBuilder.addInput(strides.asOutput());
    opBuilder.setAttr("padding", padding);
    return new MaxPoolV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new MaxPoolV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param ksize
   * @param strides
   * @param padding
   * @param options an object holding optional attributes values
   * @return a new instance of MaxPoolV2
   **/
  public static <T> MaxPoolV2<T> create(Scope scope, Operand<T> input, Operand<Integer> ksize, Operand<Integer> strides, String padding, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MaxPoolV2", scope.makeOpName("MaxPoolV2"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(ksize.asOutput());
    opBuilder.addInput(strides.asOutput());
    opBuilder.setAttr("padding", padding);
    if (options.dataFormat != null) {
      opBuilder.setAttr("dataFormat", options.dataFormat);
    }
    return new MaxPoolV2<T>(opBuilder.build());
  }
  
  /**
   * @param dataFormat
   **/
  public static Options dataFormat(String dataFormat) {
    return new Options().dataFormat(dataFormat);
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
  private MaxPoolV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
