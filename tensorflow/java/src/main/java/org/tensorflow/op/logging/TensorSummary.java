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

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class TensorSummary extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param description
     **/
    public Options description(String description) {
      this.description = description;
      return this;
    }
    
    /**
     * @param labels
     **/
    public Options labels(List<String> labels) {
      this.labels = labels;
      return this;
    }
    
    /**
     * @param displayName
     **/
    public Options displayName(String displayName) {
      this.displayName = displayName;
      return this;
    }
    
    private String description;
    private List<String> labels;
    private String displayName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TensorSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tensor
   * @return a new instance of TensorSummary
   **/
  public static <T> TensorSummary create(Scope scope, Operand<T> tensor) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorSummary", scope.makeOpName("TensorSummary"));
    opBuilder.addInput(tensor.asOutput());
    return new TensorSummary(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TensorSummary operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tensor
   * @param options an object holding optional attributes values
   * @return a new instance of TensorSummary
   **/
  public static <T> TensorSummary create(Scope scope, Operand<T> tensor, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorSummary", scope.makeOpName("TensorSummary"));
    opBuilder.addInput(tensor.asOutput());
    if (options.description != null) {
      opBuilder.setAttr("description", options.description);
    }
    if (options.labels != null) {
      opBuilder.setAttr("labels", options.labels.toArray(new String[options.labels.size()]));
    }
    if (options.displayName != null) {
      opBuilder.setAttr("displayName", options.displayName);
    }
    return new TensorSummary(opBuilder.build());
  }
  
  /**
   * @param description
   **/
  public static Options description(String description) {
    return new Options().description(description);
  }
  
  /**
   * @param labels
   **/
  public static Options labels(List<String> labels) {
    return new Options().labels(labels);
  }
  
  /**
   * @param displayName
   **/
  public static Options displayName(String displayName) {
    return new Options().displayName(displayName);
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
  private TensorSummary(Operation operation) {
    super(operation);
    int outputIdx = 0;
    summary = operation.output(outputIdx++);
  }
}
