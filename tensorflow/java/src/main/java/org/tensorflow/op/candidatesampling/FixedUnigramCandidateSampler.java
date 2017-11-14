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
package org.tensorflow.op.candidatesampling;

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class FixedUnigramCandidateSampler extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param vocabFile
     **/
    public Options vocabFile(String vocabFile) {
      this.vocabFile = vocabFile;
      return this;
    }
    
    /**
     * @param distortion
     **/
    public Options distortion(Float distortion) {
      this.distortion = distortion;
      return this;
    }
    
    /**
     * @param numReservedIds
     **/
    public Options numReservedIds(Integer numReservedIds) {
      this.numReservedIds = numReservedIds;
      return this;
    }
    
    /**
     * @param numShards
     **/
    public Options numShards(Integer numShards) {
      this.numShards = numShards;
      return this;
    }
    
    /**
     * @param shard
     **/
    public Options shard(Integer shard) {
      this.shard = shard;
      return this;
    }
    
    /**
     * @param unigrams
     **/
    public Options unigrams(List<Float> unigrams) {
      this.unigrams = unigrams;
      return this;
    }
    
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
    
    private String vocabFile;
    private Float distortion;
    private Integer numReservedIds;
    private Integer numShards;
    private Integer shard;
    private List<Float> unigrams;
    private Integer seed;
    private Integer seed2;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FixedUnigramCandidateSampler operation to the graph.
   * 
   * @param scope Current graph scope
   * @param trueClasses
   * @param numTrue
   * @param numSampled
   * @param unique
   * @param rangeMax
   * @return a new instance of FixedUnigramCandidateSampler
   **/
  public static FixedUnigramCandidateSampler create(Scope scope, Operand<Long> trueClasses, Integer numTrue, Integer numSampled, Boolean unique, Integer rangeMax) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FixedUnigramCandidateSampler", scope.makeOpName("FixedUnigramCandidateSampler"));
    opBuilder.addInput(trueClasses.asOutput());
    opBuilder.setAttr("numTrue", numTrue);
    opBuilder.setAttr("numSampled", numSampled);
    opBuilder.setAttr("unique", unique);
    opBuilder.setAttr("rangeMax", rangeMax);
    return new FixedUnigramCandidateSampler(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FixedUnigramCandidateSampler operation to the graph.
   * 
   * @param scope Current graph scope
   * @param trueClasses
   * @param numTrue
   * @param numSampled
   * @param unique
   * @param rangeMax
   * @param options an object holding optional attributes values
   * @return a new instance of FixedUnigramCandidateSampler
   **/
  public static FixedUnigramCandidateSampler create(Scope scope, Operand<Long> trueClasses, Integer numTrue, Integer numSampled, Boolean unique, Integer rangeMax, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FixedUnigramCandidateSampler", scope.makeOpName("FixedUnigramCandidateSampler"));
    opBuilder.addInput(trueClasses.asOutput());
    opBuilder.setAttr("numTrue", numTrue);
    opBuilder.setAttr("numSampled", numSampled);
    opBuilder.setAttr("unique", unique);
    opBuilder.setAttr("rangeMax", rangeMax);
    if (options.vocabFile != null) {
      opBuilder.setAttr("vocabFile", options.vocabFile);
    }
    if (options.distortion != null) {
      opBuilder.setAttr("distortion", options.distortion);
    }
    if (options.numReservedIds != null) {
      opBuilder.setAttr("numReservedIds", options.numReservedIds);
    }
    if (options.numShards != null) {
      opBuilder.setAttr("numShards", options.numShards);
    }
    if (options.shard != null) {
      opBuilder.setAttr("shard", options.shard);
    }
    if (options.unigrams != null) {
      float[] unigramsArray = new float[options.unigrams.size()];
      for (int i = 0; i < unigramsArray.length; ++i) {
        unigramsArray[i] = options.unigrams.get(i);
      }
      opBuilder.setAttr("unigrams", unigramsArray);
    }
    if (options.seed != null) {
      opBuilder.setAttr("seed", options.seed);
    }
    if (options.seed2 != null) {
      opBuilder.setAttr("seed2", options.seed2);
    }
    return new FixedUnigramCandidateSampler(opBuilder.build());
  }
  
  /**
   * @param vocabFile
   **/
  public static Options vocabFile(String vocabFile) {
    return new Options().vocabFile(vocabFile);
  }
  
  /**
   * @param distortion
   **/
  public static Options distortion(Float distortion) {
    return new Options().distortion(distortion);
  }
  
  /**
   * @param numReservedIds
   **/
  public static Options numReservedIds(Integer numReservedIds) {
    return new Options().numReservedIds(numReservedIds);
  }
  
  /**
   * @param numShards
   **/
  public static Options numShards(Integer numShards) {
    return new Options().numShards(numShards);
  }
  
  /**
   * @param shard
   **/
  public static Options shard(Integer shard) {
    return new Options().shard(shard);
  }
  
  /**
   * @param unigrams
   **/
  public static Options unigrams(List<Float> unigrams) {
    return new Options().unigrams(unigrams);
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
  
  public Output<Long> sampledCandidates() {
    return sampledCandidates;
  }
  
  public Output<Float> trueExpectedCount() {
    return trueExpectedCount;
  }
  
  public Output<Float> sampledExpectedCount() {
    return sampledExpectedCount;
  }
  
  private Output<Long> sampledCandidates;
  private Output<Float> trueExpectedCount;
  private Output<Float> sampledExpectedCount;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FixedUnigramCandidateSampler(Operation operation) {
    super(operation);
    int outputIdx = 0;
    sampledCandidates = operation.output(outputIdx++);
    trueExpectedCount = operation.output(outputIdx++);
    sampledExpectedCount = operation.output(outputIdx++);
  }
}
