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
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

public final class ImageSummary extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param maxImages
     **/
    public Options maxImages(Integer maxImages) {
      this.maxImages = maxImages;
      return this;
    }
    
    /**
     * @param badColor
     **/
    public Options badColor(Tensor<?> badColor) {
      this.badColor = badColor;
      return this;
    }
    
    private Integer maxImages;
    private Tensor<?> badColor;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ImageSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tag
   * @param tensor
   * @return a new instance of ImageSummary
   **/
  public static <T> ImageSummary create(Scope scope, Operand<String> tag, Operand<T> tensor) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ImageSummary", scope.makeOpName("ImageSummary"));
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(tensor.asOutput());
    return new ImageSummary(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ImageSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tag
   * @param tensor
   * @param options an object holding optional attributes values
   * @return a new instance of ImageSummary
   **/
  public static <T> ImageSummary create(Scope scope, Operand<String> tag, Operand<T> tensor, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ImageSummary", scope.makeOpName("ImageSummary"));
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(tensor.asOutput());
    if (options.maxImages != null) {
      opBuilder.setAttr("maxImages", options.maxImages);
    }
    if (options.badColor != null) {
      opBuilder.setAttr("badColor", options.badColor);
    }
    return new ImageSummary(opBuilder.build());
  }
  
  /**
   * @param maxImages
   **/
  public static Options maxImages(Integer maxImages) {
    return new Options().maxImages(maxImages);
  }
  
  /**
   * @param badColor
   **/
  public static Options badColor(Tensor<?> badColor) {
    return new Options().badColor(badColor);
  }
  
  public Output<String> summary() {
    return summary;
  }
  
  @Override
  public Output<String> asOutput() {
    return summary;
  }
  
  private Output<String> summary;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ImageSummary(Operation operation) {
    super(operation);
    int outputIdx = 0;
    summary = operation.output(outputIdx++);
  }
}
