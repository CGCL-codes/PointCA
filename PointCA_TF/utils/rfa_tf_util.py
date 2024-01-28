import tensorflow as tf
import  numpy as np
from data_util import  resample_pcd
from pc_distance import tf_nndistance, tf_approxmatch
from tf_ops.grouping.tf_grouping import knn_point

def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv2(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv2_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv2_%d' % (len(layer_dims) - 1))
    return outputs


def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    loss_cd_per_pc = (tf.reduce_mean(tf.sqrt(dist1), axis=1) + tf.reduce_mean(tf.sqrt(dist2), axis=1))/2
    # loss_cd_per_pc = tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1)
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    loss_cd = tf.reduce_mean(loss_cd_per_pc)
    return loss_cd


def chamfer_per_pc(pcd1, pcd2):
    """Normalised Chamfer Distance"""
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    loss_cd_per_pc = (tf.reduce_mean(tf.sqrt(dist1), axis=1) + tf.reduce_mean(tf.sqrt(dist2), axis=1))/2
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    loss_cd = tf.reduce_mean(loss_cd_per_pc)
    return loss_cd, loss_cd_per_pc


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)


def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update


def get_modified_cd_loss(pred, gt, forward_weight=1.0, threshold=None):
    """
    pred: BxNxC,
    label: BxN,
    forward_weight: relative weight for forward_distance
    """
    with tf.name_scope("cd_loss"):
        dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
        if threshold is not None:
            forward_threshold = tf.reduce_mean(dists_forward, keepdims=True, axis=1) * threshold
            backward_threshold = tf.reduce_mean(dists_backward, keepdims=True, axis=1) * threshold
            # only care about distance within threshold (ignore strong outliers)
            dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < backward_threshold, dists_backward, tf.zeros_like(dists_backward))
        # dists_forward is for each element in gt, the closest distance to this element
        dists_forward = tf.reduce_mean(dists_forward, axis=1)
        dists_backward = tf.reduce_mean(dists_backward, axis=1)
        CD_dist = forward_weight * dists_forward + dists_backward
        # CD_dist_norm = CD_dist/radius
        cd_loss = tf.reduce_mean(CD_dist)
        return cd_loss


def dense_conv(feature, growth_rate, n, k, scope, idx=None, **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=idx)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx


def dense_conv1(feature, growth_rate, n, k, scope, idx=None, **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=idx)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx


def get_edge_feature(point_cloud, k=20, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def instance_norm(net, train=True, weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keepdims=True)

    shift = tf.get_variable('shift', shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs




''' === modified from PointNet === '''

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
        data_format:   'NHWC' or 'NCHW'
    Return:
        normed:        batch-normalized maps
    """
    bn_decay = bn_decay if bn_decay is not None else 0.9
    return tf.contrib.layers.batch_norm(inputs,
                                        center=True, scale=True,
                                        is_training=is_training, decay=bn_decay, updates_collections=None,
                                        scope=scope,
                                        data_format=data_format)


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
        point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """ Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (batch_size, num_points, num_points)
        k: int

    Returns:
        nearest neighbors: (batch_size, num_points, k)
        """
    neg_adj = - adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k+1)
    return nn_idx




''' === build for PointCA === '''


def get_pc_neighbors(point_cloud, nn_idx):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]  # original batchsize
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    sess = tf.Session()
    idxx_ = idx_.eval(session=sess)
    print(idxx_)

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])  # 除了最后一维，全部展平
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors


    ''' creates the sigma-map for the batch x '''
    k = x_n.shape[-2]
    sh = x_n.shape[-1] # number of channels
    t = x_n.copy()

    pc = np.expand_dims(point_clouds_batch, axis=-2)
    pc_duplication = np.tile(pc, (1, 1, k, 1))
    pc_neigh_dist = t - pc_duplication

    radius = np.linalg.norm(pc_neigh_dist[:, :, 1:, :], axis=-1, keepdims=True)
    mean_knn_dist = np.mean(radius, axis=-2, dtype=np.float32)
    std = np.std(radius, ddof=1, axis=-2, dtype=np.float32)

    sd = mean_knn_dist + h*std

    return sd


def local_geometric_density(x_n, point_clouds_batch):
    ''' creates the sigma-map for the batch x '''
    k = x_n.shape[-2]
    sh = x_n.shape[-1] # number of channels
    t = x_n.copy()
    h = 3

    pc = np.expand_dims(point_clouds_batch, axis=-2)
    pc_duplication = np.tile(pc, (1, 1, k, 1))
    pc_neigh_dist = t - pc_duplication

    radius = np.linalg.norm(pc_neigh_dist[:, :, 1:, :], axis=-1, keepdims=True)
    mean_knn_dist = np.mean(radius, axis=-2, dtype=np.float32)
    std = np.std(radius, ddof=1, axis=-2, dtype=np.float32)
    #std = np.std(radius, axis=-2, dtype=np.float32)

    #mean = np.mean(t, axis=-2, dtype=np.float32)
    #std = np.std(t, ddof=1, axis=-2, dtype=np.float32)

    #sd = np.min(std, axis = -1)
    #sd = np.sqrt(std)
    #sd = std
    sd = mean_knn_dist + h*std

    return sd


def local_geometric_density_uniformity(x_n, point_clouds_batch,h):
    ''' creates the sigma-map for the batch x with uniformity factor'''
    k = x_n.shape[-2]
    sh = x_n.shape[-1] # number of channels
    t = x_n.copy()

    pc = np.expand_dims(point_clouds_batch, axis=-2)
    pc_duplication = np.tile(pc, (1, 1, k, 1))
    pc_neigh_dist = t - pc_duplication

    radius = np.linalg.norm(pc_neigh_dist[:, :, 1:, :], axis=-1, keepdims=True)
    mean_knn_dist = np.mean(radius, axis=-2, dtype=np.float32)
    std = np.std(radius, ddof=1, axis=-2, dtype=np.float32)

    sd = mean_knn_dist + h*std

    return sd


def channelwise_project_constraint(y, lb, ub, x_nat):

    x = np.copy(y)
    x = np.clip(x, lb, ub)
    example = np.clip(x_nat + x , -1, 1)
    x = example - x_nat

    return x


def pointwise_project_constraint(y, radius, x_nat):

    x = np.copy(y)
    example = np.clip(x_nat + x , -1, 1)
    x = example - x_nat
    norm = np.linalg.norm(x, axis = -1,keepdims =True)
    factor = np.minimum(radius / (norm + 1e-12), np.ones_like(norm))
    x = x*factor
    return x


def adaptive_project_constraint(y, sigma, eta, x_nat):

    x = np.copy(y)
    x = np.clip(x,-1,1)

    pertub = x - x_nat
    #radius = np.linalg.norm(eta*sigma, axis = -1,keepdims =True)
    radius = eta * sigma
    norm = np.linalg.norm(pertub, axis=-1,keepdims=True)
    factor = np.minimum(radius / (norm + 1e-12), np.ones_like(norm))
    pertub = pertub*factor
    example = x_nat +pertub

    return  example


def get_outlier_pc_inlier_pc(point_clouds_batch, pc_knn_neighbors, knn_dist_thresh, sor_mu):
    num_pc, num_points, _ = point_clouds_batch.shape
    k = pc_knn_neighbors.shape[-2]
    outlier_pc = []
    outlier_idx =[]
    outlier_num = np.zeros(num_pc, dtype=np.int16)
    inlier_pc = []
    removal_num = np.zeros(num_pc, dtype=np.int16)
    sor_num = np.zeros(num_pc, dtype=np.int16)

    for l in range(num_pc):
        knn_pc = pc_knn_neighbors[l]
        pc = point_clouds_batch[l]
        pc = np.expand_dims(pc, axis=-2)
        pc_duplication = np.tile(pc, (1,k,1))
        knn_distance = np.linalg.norm(pc_duplication - knn_pc, axis=-1)

        min_knn_dist = np.min(knn_distance[:,1:], axis=-1)
        outlier_idx_pc = np.where(min_knn_dist > knn_dist_thresh)[0]
        outlier_idx.append(np.copy(outlier_idx_pc))
        outlier_pc.append(np.copy(point_clouds_batch[l,outlier_idx_pc,:]))
        outlier_num[l] = len(outlier_idx_pc)

        #inlier_idx_pc = np.where(min_knn_dist <= knn_dist_thresh)[0]
        #inlier_pc.append(np.copy(point_clouds_batch[l,inlier_idx_pc,:]))

        mean_knn_dist = np.sum(knn_distance,axis=-1) / (k-1)
        outlier_removal_idx_pc = np.where(mean_knn_dist > knn_dist_thresh)[0]
        removal_num[l] = len(outlier_removal_idx_pc)

        mean_global_dist = np.mean(mean_knn_dist)
        std_global_dist = np.std(mean_knn_dist)
        statistic_removal_idx_pc = np.where(mean_knn_dist>(mean_global_dist+sor_mu*std_global_dist))[0]
        sor_num[l] = len(statistic_removal_idx_pc)

    return   outlier_num, removal_num, sor_num


def analyze_outlier_pc_inlier_pc(point_clouds_batch, pc_knn_neighbors, dist_thresh):
    num_pc, num_points, _ = point_clouds_batch.shape
    k = pc_knn_neighbors.shape[-2]
    min_outlier_pc = []
    min_outlier_num = np.zeros(num_pc, dtype=np.int16)
    min_inlier_pc = []
    min_inlier_num = np.zeros(num_pc, dtype=np.int16)

    mean_outlier_pc = []
    mean_outlier_num = np.zeros(num_pc, dtype=np.int16)
    mean_inlier_pc = []
    mean_inlier_num = np.zeros(num_pc, dtype=np.int16)

    for l in range(num_pc):
        knn_pc = pc_knn_neighbors[l]
        pc = point_clouds_batch[l]
        pc = np.expand_dims(pc, axis=-2)
        pc_duplication = np.tile(pc, (1,k,1))
        knn_distance = np.linalg.norm(pc_duplication - knn_pc, axis=-1)

        min_knn_dist = np.min(knn_distance[:,1:], axis=-1)
        min_outlier_idx_pc = np.where(min_knn_dist > dist_thresh)[0]
        #min_outlier_pc.append(np.copy(point_clouds_batch[l,min_outlier_idx_pc,:]))
        min_outlier_num[l] = len(min_outlier_idx_pc)
        min_inlier_idx_pc = np.where(min_knn_dist <= dist_thresh)[0]
        min_inlier_pc.append(np.copy(point_clouds_batch[l,min_inlier_idx_pc,:]))
        min_inlier_num[l] = len(min_inlier_idx_pc)

        mean_knn_dist = np.sum(knn_distance,axis=-1) / (k-1)
        mean_outlier_idx_pc = np.where(mean_knn_dist > dist_thresh)[0]
        #mean_outlier_pc.append(np.copy(point_clouds_batch[l, mean_outlier_idx_pc, :]))
        mean_outlier_num[l] = len(mean_outlier_idx_pc)
        mean_inlier_idx_pc = np.where(mean_knn_dist <= dist_thresh)[0]
        mean_inlier_pc.append(np.copy(point_clouds_batch[l,mean_inlier_idx_pc,:]))
        mean_inlier_num[l] = len(mean_inlier_idx_pc)

    return   min_outlier_num, min_inlier_num, min_inlier_pc, mean_outlier_num, mean_inlier_num, mean_inlier_pc


def analyze_sor_pc(point_clouds_batch, pc_knn_neighbors, sor_mu):
    num_pc, num_points, _ = point_clouds_batch.shape
    k = pc_knn_neighbors.shape[-2]

    sor_outlier_num = np.zeros(num_pc, dtype=np.int16)
    sor_inlier_pc = []
    sor_inlier_num = np.zeros(num_pc, dtype=np.int16)

    for l in range(num_pc):
        knn_pc = pc_knn_neighbors[l]
        pc = point_clouds_batch[l]
        pc = np.expand_dims(pc, axis=-2)
        pc_duplication = np.tile(pc, (1,k,1))
        knn_distance = np.linalg.norm(pc_duplication - knn_pc, axis=-1)

        mean_knn_dist = np.sum(knn_distance,axis=-1) / (k-1)
        mean_global_dist = np.mean(mean_knn_dist)
        std_global_dist = np.std(mean_knn_dist)
        sor_outlier_idx_pc = np.where(mean_knn_dist>(mean_global_dist+sor_mu*std_global_dist))[0]
        sor_outlier_num[l] = len(sor_outlier_idx_pc)
        sor_inlier_idx_pc = np.where(mean_knn_dist<=(mean_global_dist+sor_mu*std_global_dist))[0]
        sor_inlier_pc.append(np.copy(point_clouds_batch[l,sor_inlier_idx_pc,:]))
        sor_inlier_num[l] = len(sor_inlier_idx_pc)

    return   sor_outlier_num, sor_inlier_num, sor_inlier_pc


def analyze_srs_pc(point_clouds_batch, srs_ratio):
    num_pc, num_points, _ = point_clouds_batch.shape
    drop_num = int(num_points*srs_ratio)

    srs_outlier_num = np.zeros(num_pc, dtype=np.int16)
    srs_inlier_pc = []
    srs_inlier_num = np.zeros(num_pc, dtype=np.int16)

    for l in range(num_pc):

        srs_outlier_num[l] = drop_num
        srs_inlier_pc_idx = np.random.choice(num_points, num_points-drop_num, replace=False)
        srs_inlier_pc.append(np.copy(point_clouds_batch[l,srs_inlier_pc_idx,:]))
        srs_inlier_num[l] = len(srs_inlier_pc_idx)

    return   srs_outlier_num, srs_inlier_num, srs_inlier_pc