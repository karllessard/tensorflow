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

import java.util.Arrays;
import java.util.Iterator;
import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QueueDequeueUpToV2 extends PrimitiveOp implements Iterable<Operand<DataType>> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param timeoutMs
     **/
    public Options timeoutMs(Integer timeoutMs) {
      this.timeoutMs = timeoutMs;
      return this;
    }
    
    private Integer timeoutMs;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QueueDequeueUpToV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param n
   * @param componentTypes
   * @return a new instance of QueueDequeueUpToV2
   **/
  public static QueueDequeueUpToV2 create(Scope scope, Operand<?> handle, Operand<Integer> n, List<DataType> componentTypes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QueueDequeueUpToV2", scope.makeOpName("QueueDequeueUpToV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(n.asOutput());
    opBuilder.setAttr("componentTypes", componentTypes.toArray(new DataType[componentTypes.size()]));
    return new QueueDequeueUpToV2(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QueueDequeueUpToV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param n
   * @param componentTypes
   * @param options an object holding optional attributes values
   * @return a new instance of QueueDequeueUpToV2
   **/
  public static QueueDequeueUpToV2 create(Scope scope, Operand<?> handle, Operand<Integer> n, List<DataType> componentTypes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QueueDequeueUpToV2", scope.makeOpName("QueueDequeueUpToV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(n.asOutput());
    opBuilder.setAttr("componentTypes", componentTypes.toArray(new DataType[componentTypes.size()]));
    if (options.timeoutMs != null) {
      opBuilder.setAttr("timeoutMs", options.timeoutMs);
    }
    return new QueueDequeueUpToV2(opBuilder.build());
  }
  
  /**
   * @param timeoutMs
   **/
  public static Options timeoutMs(Integer timeoutMs) {
    return new Options().timeoutMs(timeoutMs);
  }
  
  public List<Output<DataType>> components() {
    return components;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<DataType>> iterator() {
    return (Iterator) components.iterator();
  }
  
  private List<Output<DataType>> components;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QueueDequeueUpToV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int componentsLength = operation.outputListLength("components");
    components = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, componentsLength));
    outputIdx += componentsLength;
  }
}
