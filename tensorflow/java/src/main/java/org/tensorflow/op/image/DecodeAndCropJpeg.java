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
import org.tensorflow.types.UInt8;

public final class DecodeAndCropJpeg extends PrimitiveOp implements Operand<UInt8> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param channels
     **/
    public Options channels(Integer channels) {
      this.channels = channels;
      return this;
    }
    
    /**
     * @param ratio
     **/
    public Options ratio(Integer ratio) {
      this.ratio = ratio;
      return this;
    }
    
    /**
     * @param fancyUpscaling
     **/
    public Options fancyUpscaling(Boolean fancyUpscaling) {
      this.fancyUpscaling = fancyUpscaling;
      return this;
    }
    
    /**
     * @param tryRecoverTruncated
     **/
    public Options tryRecoverTruncated(Boolean tryRecoverTruncated) {
      this.tryRecoverTruncated = tryRecoverTruncated;
      return this;
    }
    
    /**
     * @param acceptableFraction
     **/
    public Options acceptableFraction(Float acceptableFraction) {
      this.acceptableFraction = acceptableFraction;
      return this;
    }
    
    /**
     * @param dctMethod
     **/
    public Options dctMethod(String dctMethod) {
      this.dctMethod = dctMethod;
      return this;
    }
    
    private Integer channels;
    private Integer ratio;
    private Boolean fancyUpscaling;
    private Boolean tryRecoverTruncated;
    private Float acceptableFraction;
    private String dctMethod;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new DecodeAndCropJpeg operation to the graph.
   * 
   * @param scope Current graph scope
   * @param contents
   * @param cropWindow
   * @return a new instance of DecodeAndCropJpeg
   **/
  public static DecodeAndCropJpeg create(Scope scope, Operand<String> contents, Operand<Integer> cropWindow) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeAndCropJpeg", scope.makeOpName("DecodeAndCropJpeg"));
    opBuilder.addInput(contents.asOutput());
    opBuilder.addInput(cropWindow.asOutput());
    return new DecodeAndCropJpeg(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new DecodeAndCropJpeg operation to the graph.
   * 
   * @param scope Current graph scope
   * @param contents
   * @param cropWindow
   * @param options an object holding optional attributes values
   * @return a new instance of DecodeAndCropJpeg
   **/
  public static DecodeAndCropJpeg create(Scope scope, Operand<String> contents, Operand<Integer> cropWindow, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeAndCropJpeg", scope.makeOpName("DecodeAndCropJpeg"));
    opBuilder.addInput(contents.asOutput());
    opBuilder.addInput(cropWindow.asOutput());
    if (options.channels != null) {
      opBuilder.setAttr("channels", options.channels);
    }
    if (options.ratio != null) {
      opBuilder.setAttr("ratio", options.ratio);
    }
    if (options.fancyUpscaling != null) {
      opBuilder.setAttr("fancyUpscaling", options.fancyUpscaling);
    }
    if (options.tryRecoverTruncated != null) {
      opBuilder.setAttr("tryRecoverTruncated", options.tryRecoverTruncated);
    }
    if (options.acceptableFraction != null) {
      opBuilder.setAttr("acceptableFraction", options.acceptableFraction);
    }
    if (options.dctMethod != null) {
      opBuilder.setAttr("dctMethod", options.dctMethod);
    }
    return new DecodeAndCropJpeg(opBuilder.build());
  }
  
  /**
   * @param channels
   **/
  public static Options channels(Integer channels) {
    return new Options().channels(channels);
  }
  
  /**
   * @param ratio
   **/
  public static Options ratio(Integer ratio) {
    return new Options().ratio(ratio);
  }
  
  /**
   * @param fancyUpscaling
   **/
  public static Options fancyUpscaling(Boolean fancyUpscaling) {
    return new Options().fancyUpscaling(fancyUpscaling);
  }
  
  /**
   * @param tryRecoverTruncated
   **/
  public static Options tryRecoverTruncated(Boolean tryRecoverTruncated) {
    return new Options().tryRecoverTruncated(tryRecoverTruncated);
  }
  
  /**
   * @param acceptableFraction
   **/
  public static Options acceptableFraction(Float acceptableFraction) {
    return new Options().acceptableFraction(acceptableFraction);
  }
  
  /**
   * @param dctMethod
   **/
  public static Options dctMethod(String dctMethod) {
    return new Options().dctMethod(dctMethod);
  }
  
  public Output<UInt8> image() {
    return image;
  }
  
  @Override
  public Output<UInt8> asOutput() {
    return image;
  }
  
  private Output<UInt8> image;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private DecodeAndCropJpeg(Operation operation) {
    super(operation);
    int outputIdx = 0;
    image = operation.output(outputIdx++);
  }
}
