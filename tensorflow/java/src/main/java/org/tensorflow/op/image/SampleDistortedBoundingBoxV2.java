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

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SampleDistortedBoundingBoxV2<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param seed
     **/
    public Options seed(Integer seed) {
      this.seed = seed;
      return this;
    }
    
    /**
     * @param seed2
     **/
    public Options seed2(Integer seed2) {
      this.seed2 = seed2;
      return this;
    }
    
    /**
     * @param aspectRatioRange
     **/
    public Options aspectRatioRange(List<Float> aspectRatioRange) {
      this.aspectRatioRange = aspectRatioRange;
      return this;
    }
    
    /**
     * @param areaRange
     **/
    public Options areaRange(List<Float> areaRange) {
      this.areaRange = areaRange;
      return this;
    }
    
    /**
     * @param maxAttempts
     **/
    public Options maxAttempts(Integer maxAttempts) {
      this.maxAttempts = maxAttempts;
      return this;
    }
    
    /**
     * @param useImageIfNoBoundingBoxes
     **/
    public Options useImageIfNoBoundingBoxes(Boolean useImageIfNoBoundingBoxes) {
      this.useImageIfNoBoundingBoxes = useImageIfNoBoundingBoxes;
      return this;
    }
    
    private Integer seed;
    private Integer seed2;
    private List<Float> aspectRatioRange;
    private List<Float> areaRange;
    private Integer maxAttempts;
    private Boolean useImageIfNoBoundingBoxes;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new SampleDistortedBoundingBoxV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param imageSize
   * @param boundingBoxes
   * @param minObjectCovered
   * @return a new instance of SampleDistortedBoundingBoxV2
   **/
  public static <T> SampleDistortedBoundingBoxV2<T> create(Scope scope, Operand<T> imageSize, Operand<Float> boundingBoxes, Operand<Float> minObjectCovered) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SampleDistortedBoundingBoxV2", scope.makeOpName("SampleDistortedBoundingBoxV2"));
    opBuilder.addInput(imageSize.asOutput());
    opBuilder.addInput(boundingBoxes.asOutput());
    opBuilder.addInput(minObjectCovered.asOutput());
    return new SampleDistortedBoundingBoxV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SampleDistortedBoundingBoxV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param imageSize
   * @param boundingBoxes
   * @param minObjectCovered
   * @param options an object holding optional attributes values
   * @return a new instance of SampleDistortedBoundingBoxV2
   **/
  public static <T> SampleDistortedBoundingBoxV2<T> create(Scope scope, Operand<T> imageSize, Operand<Float> boundingBoxes, Operand<Float> minObjectCovered, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SampleDistortedBoundingBoxV2", scope.makeOpName("SampleDistortedBoundingBoxV2"));
    opBuilder.addInput(imageSize.asOutput());
    opBuilder.addInput(boundingBoxes.asOutput());
    opBuilder.addInput(minObjectCovered.asOutput());
    if (options.seed != null) {
      opBuilder.setAttr("seed", options.seed);
    }
    if (options.seed2 != null) {
      opBuilder.setAttr("seed2", options.seed2);
    }
    if (options.aspectRatioRange != null) {
      float[] aspectRatioRangeArray = new float[options.aspectRatioRange.size()];
      for (int i = 0; i < aspectRatioRangeArray.length; ++i) {
        aspectRatioRangeArray[i] = options.aspectRatioRange.get(i);
      }
      opBuilder.setAttr("aspectRatioRange", aspectRatioRangeArray);
    }
    if (options.areaRange != null) {
      float[] areaRangeArray = new float[options.areaRange.size()];
      for (int i = 0; i < areaRangeArray.length; ++i) {
        areaRangeArray[i] = options.areaRange.get(i);
      }
      opBuilder.setAttr("areaRange", areaRangeArray);
    }
    if (options.maxAttempts != null) {
      opBuilder.setAttr("maxAttempts", options.maxAttempts);
    }
    if (options.useImageIfNoBoundingBoxes != null) {
      opBuilder.setAttr("useImageIfNoBoundingBoxes", options.useImageIfNoBoundingBoxes);
    }
    return new SampleDistortedBoundingBoxV2<T>(opBuilder.build());
  }
  
  /**
   * @param seed
   **/
  public static Options seed(Integer seed) {
    return new Options().seed(seed);
  }
  
  /**
   * @param seed2
   **/
  public static Options seed2(Integer seed2) {
    return new Options().seed2(seed2);
  }
  
  /**
   * @param aspectRatioRange
   **/
  public static Options aspectRatioRange(List<Float> aspectRatioRange) {
    return new Options().aspectRatioRange(aspectRatioRange);
  }
  
  /**
   * @param areaRange
   **/
  public static Options areaRange(List<Float> areaRange) {
    return new Options().areaRange(areaRange);
  }
  
  /**
   * @param maxAttempts
   **/
  public static Options maxAttempts(Integer maxAttempts) {
    return new Options().maxAttempts(maxAttempts);
  }
  
  /**
   * @param useImageIfNoBoundingBoxes
   **/
  public static Options useImageIfNoBoundingBoxes(Boolean useImageIfNoBoundingBoxes) {
    return new Options().useImageIfNoBoundingBoxes(useImageIfNoBoundingBoxes);
  }
  
  public Output<T> begin() {
    return begin;
  }
  
  public Output<T> size() {
    return size;
  }
  
  public Output<Float> bboxes() {
    return bboxes;
  }
  
  private Output<T> begin;
  private Output<T> size;
  private Output<Float> bboxes;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SampleDistortedBoundingBoxV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    begin = operation.output(outputIdx++);
    size = operation.output(outputIdx++);
    bboxes = operation.output(outputIdx++);
  }
}
