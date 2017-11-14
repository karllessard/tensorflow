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
package org.tensorflow.op.dataflow;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class RecordInput extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param fileRandomSeed
     **/
    public Options fileRandomSeed(Integer fileRandomSeed) {
      this.fileRandomSeed = fileRandomSeed;
      return this;
    }
    
    /**
     * @param fileShuffleShiftRatio
     **/
    public Options fileShuffleShiftRatio(Float fileShuffleShiftRatio) {
      this.fileShuffleShiftRatio = fileShuffleShiftRatio;
      return this;
    }
    
    /**
     * @param fileBufferSize
     **/
    public Options fileBufferSize(Integer fileBufferSize) {
      this.fileBufferSize = fileBufferSize;
      return this;
    }
    
    /**
     * @param fileParallelism
     **/
    public Options fileParallelism(Integer fileParallelism) {
      this.fileParallelism = fileParallelism;
      return this;
    }
    
    /**
     * @param batchSize
     **/
    public Options batchSize(Integer batchSize) {
      this.batchSize = batchSize;
      return this;
    }
    
    private Integer fileRandomSeed;
    private Float fileShuffleShiftRatio;
    private Integer fileBufferSize;
    private Integer fileParallelism;
    private Integer batchSize;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new RecordInput operation to the graph.
   * 
   * @param scope Current graph scope
   * @param filePattern
   * @return a new instance of RecordInput
   **/
  public static RecordInput create(Scope scope, String filePattern) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RecordInput", scope.makeOpName("RecordInput"));
    opBuilder.setAttr("filePattern", filePattern);
    return new RecordInput(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new RecordInput operation to the graph.
   * 
   * @param scope Current graph scope
   * @param filePattern
   * @param options an object holding optional attributes values
   * @return a new instance of RecordInput
   **/
  public static RecordInput create(Scope scope, String filePattern, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RecordInput", scope.makeOpName("RecordInput"));
    opBuilder.setAttr("filePattern", filePattern);
    if (options.fileRandomSeed != null) {
      opBuilder.setAttr("fileRandomSeed", options.fileRandomSeed);
    }
    if (options.fileShuffleShiftRatio != null) {
      opBuilder.setAttr("fileShuffleShiftRatio", options.fileShuffleShiftRatio);
    }
    if (options.fileBufferSize != null) {
      opBuilder.setAttr("fileBufferSize", options.fileBufferSize);
    }
    if (options.fileParallelism != null) {
      opBuilder.setAttr("fileParallelism", options.fileParallelism);
    }
    if (options.batchSize != null) {
      opBuilder.setAttr("batchSize", options.batchSize);
    }
    return new RecordInput(opBuilder.build());
  }
  
  /**
   * @param fileRandomSeed
   **/
  public static Options fileRandomSeed(Integer fileRandomSeed) {
    return new Options().fileRandomSeed(fileRandomSeed);
  }
  
  /**
   * @param fileShuffleShiftRatio
   **/
  public static Options fileShuffleShiftRatio(Float fileShuffleShiftRatio) {
    return new Options().fileShuffleShiftRatio(fileShuffleShiftRatio);
  }
  
  /**
   * @param fileBufferSize
   **/
  public static Options fileBufferSize(Integer fileBufferSize) {
    return new Options().fileBufferSize(fileBufferSize);
  }
  
  /**
   * @param fileParallelism
   **/
  public static Options fileParallelism(Integer fileParallelism) {
    return new Options().fileParallelism(fileParallelism);
  }
  
  /**
   * @param batchSize
   **/
  public static Options batchSize(Integer batchSize) {
    return new Options().batchSize(batchSize);
  }
  
  public Output<String> records() {
    return records;
  }
  
  @Override
  public Output<String> asOutput() {
    return records;
  }
  
  private Output<String> records;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private RecordInput(Operation operation) {
    super(operation);
    int outputIdx = 0;
    records = operation.output(outputIdx++);
  }
}
