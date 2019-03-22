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

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
  
  /**
   * Controls how TensorFlow resources are cleaned up when no longer needed.
   * <p>
   * All resources allocated during an {@code EagerSession} are deleted when the session is closed.
   * <p>
   * To prevent memory leaks, resources also need to be safely deleted while the session is still active. For example, 
   * executing n operations in a loop of m iterations will allocate a minimum of n*m resources while in most cases, only 
   * resources of the last iteration are still being used.
   * <p>
   * To accomplish this task, {@code EagerSession} instances can be notified by the JVM garbage collector when TensorFlow 
   * objects are no longer being referred, so they can cleanup any native resources that are attached to them.
   */
  public static enum ResourceCleanupStrategy {
    
    /**
     * Only delete resources when the session is closed.
     * <p>
     * This is the simplest way to cleanup resources but its usage is discouraged since it can leads up to memory leaks if
     * too many resources are allocated during the session.
     */
    ON_SESSION_CLOSE,
    
    /**
     * Whenever possible, check if there are some resources that can be safely deleted, before or after completing other tasks.
     * <p>
     * Resources wont be release until a call to the TensorFlow library reaches one of this safe point. The cleanup occurs from
     * the same thread who made that call, which might cause it to block for a very short period in case there are multiple threads 
     * executing operations under the same session.
     */
    WITH_SAFE_POINTS,
    
    /**
     * Delete unused resources from a thread running in background.
     * <p>
     * This is the safest way to release unused resources, at the cost of starting and running a new thread. Each {@code EagerSession}
     * instance has its own thread, which is stopped only when the session is closed.
     * <p>
     * This strategy is used by default.
     */
    IN_BACKGROUND,
  }
  
  public static class Options {
    
    /**
     * Controls how operations dispatched are actually executed. 
     * <p>
     * When set to true, each operation are executed asynchronously (in which case some operations
     * might return "non-ready" outputs). When set to false, all operations are executed synchronously.
     * <p>
     * Synchronous execution is used by default.
     * 
     * @param value true for asynchronous execution, false for synchronous.
     */
    public Options async(boolean value) {
      async = value;
      return this;
    }
    
    /**
     * Controls how to act when we try to run an operation on a given device but
     * some input tensors are not on that device.
     * <p>
     * {@link DevicePlacementPolicy#SILENT} is used by default.
     * 
     * @param value policy to apply
     * @see {@link DevicePlacementPolicy}
     */
    public Options devicePlacementPolicy(DevicePlacementPolicy value) {
      devicePlacementPolicy = value;
      return this;
    }
    
    /**
     * Controls how TensorFlow resources are cleaned up when no longer needed.
     * <p>
     * {@link ResourceCleanupStrategy#IN_BACKGROUND} is used by default.
     * 
     * @param value strategy to use
     * @see {@link ResourceCleanupStrategy}
     */
    public Options resourceCleanupStrategy(ResourceCleanupStrategy value) {
      resourceCleanupStrategy = value;
      return this;
    }
    
    /** 
     * Builds an eager session with the selected options.
     */
    public EagerSession build() {
      return new EagerSession(async, devicePlacementPolicy, resourceCleanupStrategy);
    }
    
    private boolean async;
    private DevicePlacementPolicy devicePlacementPolicy;
    private ResourceCleanupStrategy resourceCleanupStrategy;
    
    private Options() {
      async = false;
      devicePlacementPolicy = DevicePlacementPolicy.SILENT;
      resourceCleanupStrategy = ResourceCleanupStrategy.IN_BACKGROUND;
    }
  }
  
  public static EagerSession.Options options() {
    return new Options();
  }
  
  public static EagerSession create() {
    return options().build();
  }

  private EagerSession(boolean async, DevicePlacementPolicy devicePlacementPolicy, ResourceCleanupStrategy resourceCleanupStrategy) {
    // Note: we don't support ConfigProto for now since protobuf might be discarded as valid contract between the TensorFlow 
    // clients and its core.
    this.nativeHandle = allocate(async, devicePlacementPolicy.code, null);
    this.resourceCleanupStrategy = resourceCleanupStrategy;
    if (resourceCleanupStrategy == ResourceCleanupStrategy.IN_BACKGROUND) {
      nativeResources.startBackgroundCleanup();
    }
  }

  @Override
  public synchronized void close() {
    if (nativeHandle != 0L) {
      if (resourceCleanupStrategy == ResourceCleanupStrategy.IN_BACKGROUND) {
        nativeResources.stopBackgroundCleanup();
      }
      nativeResources.deleteAll();
      delete(nativeHandle);
      nativeHandle = 0L;
    }
  }

  @Override
  public OperationBuilder opBuilder(String type, String name) {
    if (resourceCleanupStrategy == ResourceCleanupStrategy.WITH_SAFE_POINTS) {
      nativeResources.tryCleanup();
    }
    return new EagerOperationBuilder(this, createOp(nativeHandle, type), type, name);
  }
  
  /**
   * A reference to one or more allocated native resources.
   * <p>
   * Any Java objects owning native resources must declare a reference to those resources in a 
   * subclass that extends from {@code NativeReference}. When {@link NativeReference#delete()} is invoked, 
   * the resources must be freed. For example:
   * <pre>{@code
   * private static class NativeReference extends EagerSession.NativeReference {
   * 
   *    NativeReference(EagerSession session, MyClass referent, long handle) {
   *        super(session, referent);
   *        this.handle = handle;
   *    }
   *    
   *    @Override
   *    void delete() {
   *        MyClass.nativeDelete(handle);
   *    }
   *    
   *    private final long handle;
   * }
   * }</pre>
   * 
   * A Java object "owns" a native resource if this resource should not survive beyond the lifetime of
   * this object.
   * <p> 
   * <b>IMPORTANT</b>: All nested subclasses of {@code NativeReference} must be declared as static, otherwise 
   * their instances will hold an implicit reference to their parent object that might prevent them to be
   * garbage collector when they are no longer needed.
   */
  static abstract class NativeReference extends PhantomReference<Object> {

    /**
     * Attach a new phantom reference of {@code referent} to {@code session}.
     */
    public NativeReference(EagerSession session, Object referent) {
      super(referent, session.nativeResources.garbageQueue);
      nativeResources = session.nativeResources;
      nativeResources.attach(this);
    }
    
    /**
     * Detach this reference from its current session.
     * <p>
     * Clearing a NativeReference wont invoke {@link #delete()}, thus wont release the native resources it referred to. The usual
     * use case is to release the ownership of those resources by passing it to another object.
     * <p>
     * If native resources needs to be deleted as well, call {@link #delete()} explicitly.
     */
    @Override
    public void clear() {
      nativeResources.detach(this);
      super.clear();
    }
    
    /**
     * Releases all native resources owned by the referred object, now deleted.  
     */
    abstract void delete();
    
    private final NativeResourceCollector nativeResources;
  }
  
  /**
   * Keeps track of all references to native resources and clean them up when needed.
   */
  private static class NativeResourceCollector {
    
    void attach(NativeReference nativeRef) {
      synchronized(nativeRefs) {
        nativeRefs.put(nativeRef, null);
      }
    }
    
    void detach(NativeReference nativeRef) {
      synchronized(nativeRefs) {
        nativeRefs.remove(nativeRef);
      }
    }
    
    void delete(NativeReference nativeRef) {
      synchronized(nativeRefs) {
        nativeRefs.remove(nativeRef);
      }
      nativeRef.delete();
    }
    
    void deleteAll() {
      Set<NativeReference> nativeRefsToDelete;
      synchronized(nativeRefs) {
        nativeRefsToDelete = nativeRefs.keySet();
        nativeRefs.clear();
      }
      for (NativeReference nativeRef : nativeRefsToDelete) {
        nativeRef.delete();
      }
    }

    void tryCleanup() {
      Reference<?> nativeRef;
      synchronized(nativeRefs) {
        while ((nativeRef = garbageQueue.poll()) != null) {
          delete((NativeReference)nativeRef);
        }
      }
    }
    
    void startBackgroundCleanup() {
      cleanupService.execute(new Runnable() {
        @Override
        public void run() {
          try {
            while(true) {
              NativeReference nativeRef = (NativeReference)garbageQueue.remove();
              delete(nativeRef);
            }
          } catch (InterruptedException e) {
            // exit
          }
        }
      });
    }
    
    void stopBackgroundCleanup() {
      cleanupService.shutdownNow();
    }

    private final ExecutorService cleanupService = Executors.newSingleThreadExecutor();
    private final Map<NativeReference, Void> nativeRefs = new IdentityHashMap<>();
    private final ReferenceQueue<Object> garbageQueue = new ReferenceQueue<>();
  }
  
  private final NativeResourceCollector nativeResources = new NativeResourceCollector();
  private final ResourceCleanupStrategy resourceCleanupStrategy;
  private long nativeHandle;

  private static native long allocate(boolean async, int devicePlacementPolicy, byte[] config);

  private static native void delete(long handle);
  
  private static native long createOp(long contextHandle, String name);

  static {
    TensorFlow.init();
  }
}
