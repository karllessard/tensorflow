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

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class FusedResizeAndPadConv2D<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param resizeAlignCorners
     **/
    public Options resizeAlignCorners(Boolean resizeAlignCorners) {
      this.resizeAlignCorners = resizeAlignCorners;
      return this;
    }
    
    private Boolean resizeAlignCorners;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FusedResizeAndPadConv2D operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param size
   * @param paddings
   * @param filter
   * @param mode
   * @param strides
   * @param padding
   * @return a new instance of FusedResizeAndPadConv2D
   **/
  public static <T> FusedResizeAndPadConv2D<T> create(Scope scope, Operand<T> input, Operand<Integer> size, Operand<Integer> paddings, Operand<T> filter, String mode, List<Integer> strides, String padding) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FusedResizeAndPadConv2D", scope.makeOpName("FusedResizeAndPadConv2D"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(size.asOutput());
    opBuilder.addInput(paddings.asOutput());
    opBuilder.addInput(filter.asOutput());
    opBuilder.setAttr("mode", mode);
    long[] stridesArray = new long[strides.size()];
    for (int i = 0; i < stridesArray.length; ++i) {
      stridesArray[i] = strides.get(i);
    }
    opBuilder.setAttr("strides", stridesArray);
    opBuilder.setAttr("padding", padding);
    return new FusedResizeAndPadConv2D<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FusedResizeAndPadConv2D operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param size
   * @param paddings
   * @param filter
   * @param mode
   * @param strides
   * @param padding
   * @param options an object holding optional attributes values
   * @return a new instance of FusedResizeAndPadConv2D
   **/
  public static <T> FusedResizeAndPadConv2D<T> create(Scope scope, Operand<T> input, Operand<Integer> size, Operand<Integer> paddings, Operand<T> filter, String mode, List<Integer> strides, String padding, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FusedResizeAndPadConv2D", scope.makeOpName("FusedResizeAndPadConv2D"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(size.asOutput());
    opBuilder.addInput(paddings.asOutput());
    opBuilder.addInput(filter.asOutput());
    opBuilder.setAttr("mode", mode);
    long[] stridesArray = new long[strides.size()];
    for (int i = 0; i < stridesArray.length; ++i) {
      stridesArray[i] = strides.get(i);
    }
    opBuilder.setAttr("strides", stridesArray);
    opBuilder.setAttr("padding", padding);
    if (options.resizeAlignCorners != null) {
      opBuilder.setAttr("resizeAlignCorners", options.resizeAlignCorners);
    }
    return new FusedResizeAndPadConv2D<T>(opBuilder.build());
  }
  
  /**
   * @param resizeAlignCorners
   **/
  public static Options resizeAlignCorners(Boolean resizeAlignCorners) {
    return new Options().resizeAlignCorners(resizeAlignCorners);
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
  private FusedResizeAndPadConv2D(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
