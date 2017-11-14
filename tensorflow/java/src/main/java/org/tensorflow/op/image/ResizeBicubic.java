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

public final class ResizeBicubic extends PrimitiveOp implements Operand<Float> {
  
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
   * Factory method to create a class to wrap a new ResizeBicubic operation to the graph.
   * 
   * @param scope Current graph scope
   * @param images
   * @param size
   * @return a new instance of ResizeBicubic
   **/
  public static <T> ResizeBicubic create(Scope scope, Operand<T> images, Operand<Integer> size) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResizeBicubic", scope.makeOpName("ResizeBicubic"));
    opBuilder.addInput(images.asOutput());
    opBuilder.addInput(size.asOutput());
    return new ResizeBicubic(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ResizeBicubic operation to the graph.
   * 
   * @param scope Current graph scope
   * @param images
   * @param size
   * @param options an object holding optional attributes values
   * @return a new instance of ResizeBicubic
   **/
  public static <T> ResizeBicubic create(Scope scope, Operand<T> images, Operand<Integer> size, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResizeBicubic", scope.makeOpName("ResizeBicubic"));
    opBuilder.addInput(images.asOutput());
    opBuilder.addInput(size.asOutput());
    if (options.alignCorners != null) {
      opBuilder.setAttr("alignCorners", options.alignCorners);
    }
    return new ResizeBicubic(opBuilder.build());
  }
  
  /**
   * @param alignCorners
   **/
  public static Options alignCorners(Boolean alignCorners) {
    return new Options().alignCorners(alignCorners);
  }
  
  public Output<Float> resizedImages() {
    return resizedImages;
  }
  
  @Override
  public Output<Float> asOutput() {
    return resizedImages;
  }
  
  private Output<Float> resizedImages;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ResizeBicubic(Operation operation) {
    super(operation);
    int outputIdx = 0;
    resizedImages = operation.output(outputIdx++);
  }
}
