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
package org.tensorflow.op.io;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class MergeV2Checkpoints extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param deleteOldDirs
     **/
    public Options deleteOldDirs(Boolean deleteOldDirs) {
      this.deleteOldDirs = deleteOldDirs;
      return this;
    }
    
    private Boolean deleteOldDirs;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new MergeV2Checkpoints operation to the graph.
   * 
   * @param scope Current graph scope
   * @param checkpointPrefixes
   * @param destinationPrefix
   * @return a new instance of MergeV2Checkpoints
   **/
  public static MergeV2Checkpoints create(Scope scope, Operand<String> checkpointPrefixes, Operand<String> destinationPrefix) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MergeV2Checkpoints", scope.makeOpName("MergeV2Checkpoints"));
    opBuilder.addInput(checkpointPrefixes.asOutput());
    opBuilder.addInput(destinationPrefix.asOutput());
    return new MergeV2Checkpoints(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new MergeV2Checkpoints operation to the graph.
   * 
   * @param scope Current graph scope
   * @param checkpointPrefixes
   * @param destinationPrefix
   * @param options an object holding optional attributes values
   * @return a new instance of MergeV2Checkpoints
   **/
  public static MergeV2Checkpoints create(Scope scope, Operand<String> checkpointPrefixes, Operand<String> destinationPrefix, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MergeV2Checkpoints", scope.makeOpName("MergeV2Checkpoints"));
    opBuilder.addInput(checkpointPrefixes.asOutput());
    opBuilder.addInput(destinationPrefix.asOutput());
    if (options.deleteOldDirs != null) {
      opBuilder.setAttr("deleteOldDirs", options.deleteOldDirs);
    }
    return new MergeV2Checkpoints(opBuilder.build());
  }
  
  /**
   * @param deleteOldDirs
   **/
  public static Options deleteOldDirs(Boolean deleteOldDirs) {
    return new Options().deleteOldDirs(deleteOldDirs);
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private MergeV2Checkpoints(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
