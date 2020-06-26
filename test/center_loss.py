import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops



def self_center(data, segment_ids, num_segments,name=None):
    segment_ids=tf.reshape(segment_ids,[-1])
    # a=tf.where(tf.equal(segment_ids,1))
    # a=tf.reshape(a,[-1,])
    center1=tf.gather(data,tf.reshape(tf.where(tf.equal(segment_ids,0)),[-1,]))
    center0=tf.gather(data,tf.reshape(tf.where(tf.equal(segment_ids,1)),[-1,]))
    t1=tf.reduce_mean(center1,axis=0,keep_dims=True)
    t2=tf.reduce_mean(center0,axis=0,keep_dims=True)
    return tf.concat([t1,t2],axis=0)


def _unsorted_segment_N(data, segment_ids, num_segments):
  """ Helper function for unsorted_segment_mean/_sqrtN. Computes the number
      of segment entries with 0-entries set to 1 to allow division by N.
  """
  # bincount doesn't support negative indices so we use unsorted_segment_sum
  segment_ids_shape = array_ops.shape_internal(segment_ids)
  ones_tensor = array_ops.ones(segment_ids_shape, dtype=data.dtype)
  N = gen_math_ops.unsorted_segment_sum(ones_tensor, segment_ids, num_segments)
  # add dimensions for all non-reduced axes
  ndims_output = data.shape.ndims - segment_ids.shape.ndims
  broadcast_shape = [num_segments] + [1] * ndims_output
  N = array_ops.reshape(N, broadcast_shape)
  return gen_math_ops.maximum(N, 1)

def unsorted_segment_mean(data, segment_ids, num_segments,name=None):
    with ops.name_scope(name, "UnsortedSegmentMean"):
        data = ops.convert_to_tensor(data)
        segment_ids = ops.convert_to_tensor(segment_ids)
        N = _unsorted_segment_N(data, segment_ids, num_segments)
        summed = gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments)
        return summed / N


def self_center(data, segment_ids, num_segments,name=None):
    segment_ids=tf.reshape(segment_ids,[-1])
    # a=tf.where(tf.equal(segment_ids,1))
    # a=tf.reshape(a,[-1,])
    center1=tf.gather(data,tf.reshape(tf.where(tf.equal(segment_ids,0)),[-1,]))
    center0=tf.gather(data,tf.reshape(tf.where(tf.equal(segment_ids,1)),[-1,]))
    t1=tf.reduce_mean(center1,axis=0,keep_dims=True)
    t2=tf.reduce_mean(center0,axis=0,keep_dims=True)
    return tf.concat([t1,t2],axis=0)
def get_center_loss(features, labels, num_classes):

    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, dtype=tf.int32)
##############################################################
    centers0=unsorted_segment_mean(features,labels,num_classes)
    EdgeWeights=tf.ones((num_classes,num_classes))-tf.eye(num_classes)
    margin=tf.constant(0.9,dtype="float32")
    margin1=tf.constant(0.2,dtype="float32")
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    center_pairwise_dist = tf.transpose(norm(tf.expand_dims(centers0, 2) - tf.transpose(centers0)))
    loss_0= tf.reduce_sum(tf.multiply(tf.maximum(0.0, margin-center_pairwise_dist),EdgeWeights))
###########################################################################
    centers_batch = tf.gather(centers0, labels)
    loss_1 = tf.maximum(0.0,tf.nn.l2_loss(features - centers_batch)- margin1)

    return (loss_0 + loss_1)/(2*10+2*2)
