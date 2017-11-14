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
package org.tensorflow.op.controlflow;

import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Abort extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param errorMsg
     **/
    public Options errorMsg(String errorMsg) {
      this.errorMsg = errorMsg;
      return this;
    }
    
    /**
     * @param exitWithoutError
     **/
    public Options exitWithoutError(Boolean exitWithoutError) {
      this.exitWithoutError = exitWithoutError;
      return this;
    }
    
    private String errorMsg;
    private Boolean exitWithoutError;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Abort operation to the graph.
   * 
   * @param scope Current graph scope
   * @return a new instance of Abort
   **/
  public static Abort create(Scope scope) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Abort", scope.makeOpName("Abort"));
    return new Abort(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Abort operation to the graph.
   * 
   * @param scope Current graph scope
   * @param options an object holding optional attributes values
   * @return a new instance of Abort
   **/
  public static Abort create(Scope scope, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Abort", scope.makeOpName("Abort"));
    if (options.errorMsg != null) {
      opBuilder.setAttr("errorMsg", options.errorMsg);
    }
    if (options.exitWithoutError != null) {
      opBuilder.setAttr("exitWithoutError", options.exitWithoutError);
    }
    return new Abort(opBuilder.build());
  }
  
  /**
   * @param errorMsg
   **/
  public static Options errorMsg(String errorMsg) {
    return new Options().errorMsg(errorMsg);
  }
  
  /**
   * @param exitWithoutError
   **/
  public static Options exitWithoutError(Boolean exitWithoutError) {
    return new Options().exitWithoutError(exitWithoutError);
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Abort(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
