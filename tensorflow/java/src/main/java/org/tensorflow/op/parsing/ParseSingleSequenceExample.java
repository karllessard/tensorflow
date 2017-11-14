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
package org.tensorflow.op.parsing;

import java.util.Arrays;
import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class ParseSingleSequenceExample extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param contextSparseTypes
     **/
    public Options contextSparseTypes(List<DataType> contextSparseTypes) {
      this.contextSparseTypes = contextSparseTypes;
      return this;
    }
    
    /**
     * @param featureListDenseTypes
     **/
    public Options featureListDenseTypes(List<DataType> featureListDenseTypes) {
      this.featureListDenseTypes = featureListDenseTypes;
      return this;
    }
    
    /**
     * @param contextDenseShapes
     **/
    public Options contextDenseShapes(List<Shape> contextDenseShapes) {
      this.contextDenseShapes = contextDenseShapes;
      return this;
    }
    
    /**
     * @param featureListSparseTypes
     **/
    public Options featureListSparseTypes(List<DataType> featureListSparseTypes) {
      this.featureListSparseTypes = featureListSparseTypes;
      return this;
    }
    
    /**
     * @param featureListDenseShapes
     **/
    public Options featureListDenseShapes(List<Shape> featureListDenseShapes) {
      this.featureListDenseShapes = featureListDenseShapes;
      return this;
    }
    
    private List<DataType> contextSparseTypes;
    private List<DataType> featureListDenseTypes;
    private List<Shape> contextDenseShapes;
    private List<DataType> featureListSparseTypes;
    private List<Shape> featureListDenseShapes;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ParseSingleSequenceExample operation to the graph.
   * 
   * @param scope Current graph scope
   * @param serialized
   * @param featureListDenseMissingAssumedEmpty
   * @param contextSparseKeys
   * @param contextDenseKeys
   * @param featureListSparseKeys
   * @param featureListDenseKeys
   * @param contextDenseDefaults
   * @param debugName
   * @return a new instance of ParseSingleSequenceExample
   **/
  public static ParseSingleSequenceExample create(Scope scope, Operand<String> serialized, Operand<String> featureListDenseMissingAssumedEmpty, Iterable<Operand<String>> contextSparseKeys, Iterable<Operand<String>> contextDenseKeys, Iterable<Operand<String>> featureListSparseKeys, Iterable<Operand<String>> featureListDenseKeys, Iterable<Operand<?>> contextDenseDefaults, Operand<String> debugName) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ParseSingleSequenceExample", scope.makeOpName("ParseSingleSequenceExample"));
    opBuilder.addInput(serialized.asOutput());
    opBuilder.addInput(featureListDenseMissingAssumedEmpty.asOutput());
    opBuilder.addInputList(Operands.asOutputs(contextSparseKeys));
    opBuilder.addInputList(Operands.asOutputs(contextDenseKeys));
    opBuilder.addInputList(Operands.asOutputs(featureListSparseKeys));
    opBuilder.addInputList(Operands.asOutputs(featureListDenseKeys));
    opBuilder.addInputList(Operands.asOutputs(contextDenseDefaults));
    opBuilder.addInput(debugName.asOutput());
    return new ParseSingleSequenceExample(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ParseSingleSequenceExample operation to the graph.
   * 
   * @param scope Current graph scope
   * @param serialized
   * @param featureListDenseMissingAssumedEmpty
   * @param contextSparseKeys
   * @param contextDenseKeys
   * @param featureListSparseKeys
   * @param featureListDenseKeys
   * @param contextDenseDefaults
   * @param debugName
   * @param options an object holding optional attributes values
   * @return a new instance of ParseSingleSequenceExample
   **/
  public static ParseSingleSequenceExample create(Scope scope, Operand<String> serialized, Operand<String> featureListDenseMissingAssumedEmpty, Iterable<Operand<String>> contextSparseKeys, Iterable<Operand<String>> contextDenseKeys, Iterable<Operand<String>> featureListSparseKeys, Iterable<Operand<String>> featureListDenseKeys, Iterable<Operand<?>> contextDenseDefaults, Operand<String> debugName, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ParseSingleSequenceExample", scope.makeOpName("ParseSingleSequenceExample"));
    opBuilder.addInput(serialized.asOutput());
    opBuilder.addInput(featureListDenseMissingAssumedEmpty.asOutput());
    opBuilder.addInputList(Operands.asOutputs(contextSparseKeys));
    opBuilder.addInputList(Operands.asOutputs(contextDenseKeys));
    opBuilder.addInputList(Operands.asOutputs(featureListSparseKeys));
    opBuilder.addInputList(Operands.asOutputs(featureListDenseKeys));
    opBuilder.addInputList(Operands.asOutputs(contextDenseDefaults));
    opBuilder.addInput(debugName.asOutput());
    if (options.contextSparseTypes != null) {
      opBuilder.setAttr("contextSparseTypes", options.contextSparseTypes.toArray(new DataType[options.contextSparseTypes.size()]));
    }
    if (options.featureListDenseTypes != null) {
      opBuilder.setAttr("featureListDenseTypes", options.featureListDenseTypes.toArray(new DataType[options.featureListDenseTypes.size()]));
    }
    if (options.contextDenseShapes != null) {
      opBuilder.setAttr("contextDenseShapes", options.contextDenseShapes.toArray(new Shape[options.contextDenseShapes.size()]));
    }
    if (options.featureListSparseTypes != null) {
      opBuilder.setAttr("featureListSparseTypes", options.featureListSparseTypes.toArray(new DataType[options.featureListSparseTypes.size()]));
    }
    if (options.featureListDenseShapes != null) {
      opBuilder.setAttr("featureListDenseShapes", options.featureListDenseShapes.toArray(new Shape[options.featureListDenseShapes.size()]));
    }
    return new ParseSingleSequenceExample(opBuilder.build());
  }
  
  /**
   * @param contextSparseTypes
   **/
  public static Options contextSparseTypes(List<DataType> contextSparseTypes) {
    return new Options().contextSparseTypes(contextSparseTypes);
  }
  
  /**
   * @param featureListDenseTypes
   **/
  public static Options featureListDenseTypes(List<DataType> featureListDenseTypes) {
    return new Options().featureListDenseTypes(featureListDenseTypes);
  }
  
  /**
   * @param contextDenseShapes
   **/
  public static Options contextDenseShapes(List<Shape> contextDenseShapes) {
    return new Options().contextDenseShapes(contextDenseShapes);
  }
  
  /**
   * @param featureListSparseTypes
   **/
  public static Options featureListSparseTypes(List<DataType> featureListSparseTypes) {
    return new Options().featureListSparseTypes(featureListSparseTypes);
  }
  
  /**
   * @param featureListDenseShapes
   **/
  public static Options featureListDenseShapes(List<Shape> featureListDenseShapes) {
    return new Options().featureListDenseShapes(featureListDenseShapes);
  }
  
  public List<Output<Long>> contextSparseIndices() {
    return contextSparseIndices;
  }
  
  public List<Output<DataType>> contextSparseValues() {
    return contextSparseValues;
  }
  
  public List<Output<Long>> contextSparseShapes() {
    return contextSparseShapes;
  }
  
  public List<Output<?>> contextDenseValues() {
    return contextDenseValues;
  }
  
  public List<Output<Long>> featureListSparseIndices() {
    return featureListSparseIndices;
  }
  
  public List<Output<DataType>> featureListSparseValues() {
    return featureListSparseValues;
  }
  
  public List<Output<Long>> featureListSparseShapes() {
    return featureListSparseShapes;
  }
  
  public List<Output<DataType>> featureListDenseValues() {
    return featureListDenseValues;
  }
  
  private List<Output<Long>> contextSparseIndices;
  private List<Output<DataType>> contextSparseValues;
  private List<Output<Long>> contextSparseShapes;
  private List<Output<?>> contextDenseValues;
  private List<Output<Long>> featureListSparseIndices;
  private List<Output<DataType>> featureListSparseValues;
  private List<Output<Long>> featureListSparseShapes;
  private List<Output<DataType>> featureListDenseValues;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ParseSingleSequenceExample(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int contextSparseIndicesLength = operation.outputListLength("contextSparseIndices");
    contextSparseIndices = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, contextSparseIndicesLength));
    outputIdx += contextSparseIndicesLength;
    int contextSparseValuesLength = operation.outputListLength("contextSparseValues");
    contextSparseValues = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, contextSparseValuesLength));
    outputIdx += contextSparseValuesLength;
    int contextSparseShapesLength = operation.outputListLength("contextSparseShapes");
    contextSparseShapes = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, contextSparseShapesLength));
    outputIdx += contextSparseShapesLength;
    int contextDenseValuesLength = operation.outputListLength("contextDenseValues");
    contextDenseValues = Arrays.asList(operation.outputList(outputIdx, contextDenseValuesLength));
    outputIdx += contextDenseValuesLength;
    int featureListSparseIndicesLength = operation.outputListLength("featureListSparseIndices");
    featureListSparseIndices = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, featureListSparseIndicesLength));
    outputIdx += featureListSparseIndicesLength;
    int featureListSparseValuesLength = operation.outputListLength("featureListSparseValues");
    featureListSparseValues = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, featureListSparseValuesLength));
    outputIdx += featureListSparseValuesLength;
    int featureListSparseShapesLength = operation.outputListLength("featureListSparseShapes");
    featureListSparseShapes = Arrays.asList((Output<Long>[])operation.outputList(outputIdx, featureListSparseShapesLength));
    outputIdx += featureListSparseShapesLength;
    int featureListDenseValuesLength = operation.outputListLength("featureListDenseValues");
    featureListDenseValues = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, featureListDenseValuesLength));
    outputIdx += featureListDenseValuesLength;
  }
}
