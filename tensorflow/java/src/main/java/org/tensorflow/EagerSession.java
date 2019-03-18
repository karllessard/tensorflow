/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
package org.tensorflow;

import java.io.Closeable;
import java.util.HashSet;
import java.util.Set;

public final class EagerSession implements ExecutionEnvironment, AutoCloseable {

  /**
   * Controls how to act when we try to run an operation on a given device but
   * some input tensors are not on that device.
   */
  public static enum DevicePlacementPolicy {

    /** 
     * Running operations with input tensors on the wrong device will fail. 
     */
    EXPLICIT(0),

    /** 
     * Copy the tensor to the right device but log a warning. 
     */
    WARN(1),

    /** 
     * Silently copy the tensor, which has a performance cost since the 
     * operation will be blocked till the copy completes. This is the default 
     * placement policy.
     */
    SILENT(2),

    /** 
     * Placement policy which silently copies int32 tensors but not other 
     * dtypes.
     */
    SILENT_FOR_INT32(3);
    
    final int code;
    
    private DevicePlacementPolicy(int code) {
      this.code = code;
    }
  }
  
  public static class Options {
    
    public Options async(boolean value) {
      async = value;
      return this;
    }
    
    public Options devicePlacementPolicy(DevicePlacementPolicy value) {
      devicePlacementPolicy = value;
      return this;
    }
    
    public Options config(byte[] value) {
      config = value;
      return this;
    }
    
    public EagerSession build() {
      return new EagerSession(async, devicePlacementPolicy, config);
    }
    
    private boolean async;
    private DevicePlacementPolicy devicePlacementPolicy;
    private byte[] config;
    
    private Options() {
      async = false;
      devicePlacementPolicy = DevicePlacementPolicy.SILENT;
      config = null;
    }
  }
  
  public static EagerSession.Options options() {
    return new Options();
  }
  
  public static EagerSession create() {
    return options().build();
  }

  private EagerSession(boolean async, DevicePlacementPolicy devicePlacementPolicy, byte[] config) {
    this.nativeHandle = allocate(async, devicePlacementPolicy.code, config);
    this.allocatedResources = new ResourceHeap();
  }

  @Override
  public OperationBuilder opBuilder(String type, String name) {
    EagerOperationBuilder builder = new EagerOperationBuilder(this, type, name);
    allocatedResources.add(builder);
    return builder;
  }

  @Override
  public void close() {
    // First delete any resource we might have allocated during this session
    // (e.g. ops, tensor handles...)
    allocatedResources.closeAll();

    if (nativeHandle != 0) {
      delete(nativeHandle);
      nativeHandle = 0;
    }
  }
  
  // TODO synchronize?
  static class ResourceHeap {
    
    void add(Tensor<?> tensor) {
      tensor.takeOwnership();
      resources.add(tensor);
    }

    void add(Closeable resource) {
      resources.add(resource); 
    }
    
    void remove(Tensor<?> tensor) {
      tensor.releaseOwnership();
      resources.remove(tensor);
    }

    void remove(Closeable resource) {
      resources.remove(resource);
    }
    
    private void closeAll() {
      for (Closeable resource : resources) {
        resource.close();
      }
      resources.clear();
    }

    private Set<Closeable> resources = new HashSet<>();
  }
  
  ResourceHeap allocatedResources() {
    return allocatedResources;
  }
  
  long nativeHandle() {
    return nativeHandle;
  }
  
  private ResourceHeap allocatedResources;
  private long nativeHandle;

  private static native long allocate(boolean async, int devicePlacementPolicy,
      byte[] config);

  private static native void delete(long handle);

  static {
    TensorFlow.init();
  }
}
