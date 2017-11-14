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

public final class EncodeJpeg extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param format
     **/
    public Options format(String format) {
      this.format = format;
      return this;
    }
    
    /**
     * @param quality
     **/
    public Options quality(Integer quality) {
      this.quality = quality;
      return this;
    }
    
    /**
     * @param progressive
     **/
    public Options progressive(Boolean progressive) {
      this.progressive = progressive;
      return this;
    }
    
    /**
     * @param optimizeSize
     **/
    public Options optimizeSize(Boolean optimizeSize) {
      this.optimizeSize = optimizeSize;
      return this;
    }
    
    /**
     * @param chromaDownsampling
     **/
    public Options chromaDownsampling(Boolean chromaDownsampling) {
      this.chromaDownsampling = chromaDownsampling;
      return this;
    }
    
    /**
     * @param densityUnit
     **/
    public Options densityUnit(String densityUnit) {
      this.densityUnit = densityUnit;
      return this;
    }
    
    /**
     * @param xDensity
     **/
    public Options xDensity(Integer xDensity) {
      this.xDensity = xDensity;
      return this;
    }
    
    /**
     * @param yDensity
     **/
    public Options yDensity(Integer yDensity) {
      this.yDensity = yDensity;
      return this;
    }
    
    /**
     * @param xmpMetadata
     **/
    public Options xmpMetadata(String xmpMetadata) {
      this.xmpMetadata = xmpMetadata;
      return this;
    }
    
    private String format;
    private Integer quality;
    private Boolean progressive;
    private Boolean optimizeSize;
    private Boolean chromaDownsampling;
    private String densityUnit;
    private Integer xDensity;
    private Integer yDensity;
    private String xmpMetadata;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new EncodeJpeg operation to the graph.
   * 
   * @param scope Current graph scope
   * @param image
   * @return a new instance of EncodeJpeg
   **/
  public static EncodeJpeg create(Scope scope, Operand<UInt8> image) {
    OperationBuilder opBuilder = scope.graph().opBuilder("EncodeJpeg", scope.makeOpName("EncodeJpeg"));
    opBuilder.addInput(image.asOutput());
    return new EncodeJpeg(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new EncodeJpeg operation to the graph.
   * 
   * @param scope Current graph scope
   * @param image
   * @param options an object holding optional attributes values
   * @return a new instance of EncodeJpeg
   **/
  public static EncodeJpeg create(Scope scope, Operand<UInt8> image, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("EncodeJpeg", scope.makeOpName("EncodeJpeg"));
    opBuilder.addInput(image.asOutput());
    if (options.format != null) {
      opBuilder.setAttr("format", options.format);
    }
    if (options.quality != null) {
      opBuilder.setAttr("quality", options.quality);
    }
    if (options.progressive != null) {
      opBuilder.setAttr("progressive", options.progressive);
    }
    if (options.optimizeSize != null) {
      opBuilder.setAttr("optimizeSize", options.optimizeSize);
    }
    if (options.chromaDownsampling != null) {
      opBuilder.setAttr("chromaDownsampling", options.chromaDownsampling);
    }
    if (options.densityUnit != null) {
      opBuilder.setAttr("densityUnit", options.densityUnit);
    }
    if (options.xDensity != null) {
      opBuilder.setAttr("xDensity", options.xDensity);
    }
    if (options.yDensity != null) {
      opBuilder.setAttr("yDensity", options.yDensity);
    }
    if (options.xmpMetadata != null) {
      opBuilder.setAttr("xmpMetadata", options.xmpMetadata);
    }
    return new EncodeJpeg(opBuilder.build());
  }
  
  /**
   * @param format
   **/
  public static Options format(String format) {
    return new Options().format(format);
  }
  
  /**
   * @param quality
   **/
  public static Options quality(Integer quality) {
    return new Options().quality(quality);
  }
  
  /**
   * @param progressive
   **/
  public static Options progressive(Boolean progressive) {
    return new Options().progressive(progressive);
  }
  
  /**
   * @param optimizeSize
   **/
  public static Options optimizeSize(Boolean optimizeSize) {
    return new Options().optimizeSize(optimizeSize);
  }
  
  /**
   * @param chromaDownsampling
   **/
  public static Options chromaDownsampling(Boolean chromaDownsampling) {
    return new Options().chromaDownsampling(chromaDownsampling);
  }
  
  /**
   * @param densityUnit
   **/
  public static Options densityUnit(String densityUnit) {
    return new Options().densityUnit(densityUnit);
  }
  
  /**
   * @param xDensity
   **/
  public static Options xDensity(Integer xDensity) {
    return new Options().xDensity(xDensity);
  }
  
  /**
   * @param yDensity
   **/
  public static Options yDensity(Integer yDensity) {
    return new Options().yDensity(yDensity);
  }
  
  /**
   * @param xmpMetadata
   **/
  public static Options xmpMetadata(String xmpMetadata) {
    return new Options().xmpMetadata(xmpMetadata);
  }
  
  public Output<String> contents() {
    return contents;
  }
  
  @Override
  public Output<String> asOutput() {
    return contents;
  }
  
  private Output<String> contents;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private EncodeJpeg(Operation operation) {
    super(operation);
    int outputIdx = 0;
    contents = operation.output(outputIdx++);
  }
}
