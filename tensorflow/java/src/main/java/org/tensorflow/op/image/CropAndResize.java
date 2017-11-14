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

public final class CropAndResize extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param method
     **/
    public Options method(String method) {
      this.method = method;
      return this;
    }
    
    /**
     * @param extrapolationValue
     **/
    public Options extrapolationValue(Float extrapolationValue) {
      this.extrapolationValue = extrapolationValue;
      return this;
    }
    
    private String method;
    private Float extrapolationValue;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new CropAndResize operation to the graph.
   * 
   * @param scope Current graph scope
   * @param image
   * @param boxes
   * @param boxInd
   * @param cropSize
   * @return a new instance of CropAndResize
   **/
  public static <T> CropAndResize create(Scope scope, Operand<T> image, Operand<Float> boxes, Operand<Integer> boxInd, Operand<Integer> cropSize) {
    OperationBuilder opBuilder = scope.graph().opBuilder("CropAndResize", scope.makeOpName("CropAndResize"));
    opBuilder.addInput(image.asOutput());
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(boxInd.asOutput());
    opBuilder.addInput(cropSize.asOutput());
    return new CropAndResize(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new CropAndResize operation to the graph.
   * 
   * @param scope Current graph scope
   * @param image
   * @param boxes
   * @param boxInd
   * @param cropSize
   * @param options an object holding optional attributes values
   * @return a new instance of CropAndResize
   **/
  public static <T> CropAndResize create(Scope scope, Operand<T> image, Operand<Float> boxes, Operand<Integer> boxInd, Operand<Integer> cropSize, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("CropAndResize", scope.makeOpName("CropAndResize"));
    opBuilder.addInput(image.asOutput());
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(boxInd.asOutput());
    opBuilder.addInput(cropSize.asOutput());
    if (options.method != null) {
      opBuilder.setAttr("method", options.method);
    }
    if (options.extrapolationValue != null) {
      opBuilder.setAttr("extrapolationValue", options.extrapolationValue);
    }
    return new CropAndResize(opBuilder.build());
  }
  
  /**
   * @param method
   **/
  public static Options method(String method) {
    return new Options().method(method);
  }
  
  /**
   * @param extrapolationValue
   **/
  public static Options extrapolationValue(Float extrapolationValue) {
    return new Options().extrapolationValue(extrapolationValue);
  }
  
  public Output<Float> crops() {
    return crops;
  }
  
  @Override
  public Output<Float> asOutput() {
    return crops;
  }
  
  private Output<Float> crops;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private CropAndResize(Operation operation) {
    super(operation);
    int outputIdx = 0;
    crops = operation.output(outputIdx++);
  }
}
