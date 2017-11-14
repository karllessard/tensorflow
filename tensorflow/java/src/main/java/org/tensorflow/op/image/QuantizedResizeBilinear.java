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
package org.tensorflow.op.image;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizedResizeBilinear<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param alignCorners
     **/
    public Options alignCorners(Boolean alignCorners) {
      this.alignCorners = alignCorners;
      return this;
    }
    
    private Boolean alignCorners;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizedResizeBilinear operation to the graph.
   * 
   * @param scope Current graph scope
   * @param images
   * @param size
   * @param min
   * @param max
   * @return a new instance of QuantizedResizeBilinear
   **/
  public static <T> QuantizedResizeBilinear<T> create(Scope scope, Operand<T> images, Operand<Integer> size, Operand<Float> min, Operand<Float> max) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedResizeBilinear", scope.makeOpName("QuantizedResizeBilinear"));
    opBuilder.addInput(images.asOutput());
    opBuilder.addInput(size.asOutput());
    opBuilder.addInput(min.asOutput());
    opBuilder.addInput(max.asOutput());
    return new QuantizedResizeBilinear<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizedResizeBilinear operation to the graph.
   * 
   * @param scope Current graph scope
   * @param images
   * @param size
   * @param min
   * @param max
   * @param options an object holding optional attributes values
   * @return a new instance of QuantizedResizeBilinear
   **/
  public static <T> QuantizedResizeBilinear<T> create(Scope scope, Operand<T> images, Operand<Integer> size, Operand<Float> min, Operand<Float> max, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedResizeBilinear", scope.makeOpName("QuantizedResizeBilinear"));
    opBuilder.addInput(images.asOutput());
    opBuilder.addInput(size.asOutput());
    opBuilder.addInput(min.asOutput());
    opBuilder.addInput(max.asOutput());
    if (options.alignCorners != null) {
      opBuilder.setAttr("alignCorners", options.alignCorners);
    }
    return new QuantizedResizeBilinear<T>(opBuilder.build());
  }
  
  /**
   * @param alignCorners
   **/
  public static Options alignCorners(Boolean alignCorners) {
    return new Options().alignCorners(alignCorners);
  }
  
  public Output<T> resizedImages() {
    return resizedImages;
  }
  
  public Output<Float> outMin() {
    return outMin;
  }
  
  public Output<Float> outMax() {
    return outMax;
  }
  
  private Output<T> resizedImages;
  private Output<Float> outMin;
  private Output<Float> outMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedResizeBilinear(Operation operation) {
    super(operation);
    int outputIdx = 0;
    resizedImages = operation.output(outputIdx++);
    outMin = operation.output(outputIdx++);
    outMax = operation.output(outputIdx++);
  }
}
