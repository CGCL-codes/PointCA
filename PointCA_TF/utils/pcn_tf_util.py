import tensorflow as tf
import  numpy as np
from pc_distance import tf_nndistance, tf_approxmatch
'''
try:
    from pc_distance import tf_nndistance, tf_approxmatch
except:
    pass
'''


'''mlp and conv1d with stride 1 are different'''


def mlp(features, layer_dims, bn=None, bn_params=None):
    # doc: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/fully_connected
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
    # doc: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/conv1d
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    # kernel size -> single value for all spatial dimensions
    # the size of filter should be (1, 3)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def point_maxpool(inputs, npts, keepdims=False):
    # number of points, number of channels -> get the maximum value along the number of channels
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims) for f in tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i, f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)


def chamfer(pcd1, pcd2):
    """Normalised Chamfer Distance"""
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    loss_cd_per_pc = (tf.reduce_mean(tf.sqrt(dist1), axis=1) + tf.reduce_mean(tf.sqrt(dist2), axis=1))/2
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    loss_cd = tf.reduce_mean(loss_cd_per_pc)    # 对一个batch
    return loss_cd, loss_cd_per_pc


def chamfer_partial_inputs(pcd1, pcd2, batchsize):
    """Normalised Chamfer Distance"""
    per_pc1 = tf.squeeze(tf.split(pcd1, batchsize, axis=1))
    per_pc2 = tf.squeeze(tf.split(pcd2, batchsize, axis=1))
    dist1, _, dist2, _ = tf_nndistance.nn_distance(per_pc1, per_pc2)
    loss_cd_per_pc = (tf.reduce_mean(tf.sqrt(dist1), axis=1) + tf.reduce_mean(tf.sqrt(dist2), axis=1))/2
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    loss_cd = tf.reduce_mean(loss_cd_per_pc)
    return loss_cd, loss_cd_per_pc


def earth_mover(pcd1, pcd2):
    """Normalised Earth Mover Distance"""
    assert pcd1.shape[1] == pcd2.shape[1]  # has the same number of points
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)


def earth_mover_per_pc(pcd1, pcd2):
    """Normalised Earth Mover Distance"""
    assert pcd1.shape[1] == pcd2.shape[1]  # has the same number of points
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return  match, cost, cost / num_points, tf.reduce_mean(cost / num_points)


def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update




''' === borrow from PointNet === '''


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        use_fp16: use 16 bit float or 32 bit float
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
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


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Variable, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controlling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')  # basically just mean and variance
        
        decay = bn_decay if bn_decay is not None else 0.9


        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
        '''tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)'''
    # 这里的beta, gamma不是3rd/4th moment, 是transferred mean and variance
    # y_i = gamma * x_i + beta, 其中 x_i 是 normalized 之后的结果
    # ref: https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Variable, true indicates training phase
        bn_decay:    float or float tensor variable, controlling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bias=True,
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
        if bias:
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
        '''
        tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC',
                        name=None, input=None)
        value: (NHWC) -> Number of Batch * In Height * In Width * In Channel
        kzise:

        '''
        return outputs


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bias=True,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC (Batch Size * Height * Width * Channel)
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true,
                    xavier initializer is the weights initialization technique
                    that tries to make the variance of the outputs of a layer
                    to be equal to the variance of its inputs
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bias: bool, whether to add bias or not
      bn_decay: float or float tensor variable in [0,1] -> actually no idea = =
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        # either [1, 1] or [1, 3]
        kernel_h, kernel_w = kernel_size
        # 64, 128, 256, 512, 1028
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]

        # not using weight_dacay, since we are using xavier initializer,
        # so stddev is not used since it is the setting for truncated_normal_initializer()
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        # always [1, 1]
        stride_h, stride_w = stride

        # tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)
        # filters -> [filter_height, filter_width, in_channels, out_channels], [1,1,1,1] or [1,1,3,1]
        # -> Point-Based MLPs
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        if bias:
            biases = _variable_on_cpu('biases', [num_output_channels], tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

        # always use batch normalisation
        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')

        # always use relu activation function
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps. """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


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


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
        """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    # edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    edge_feature = tf.concat([point_cloud_neighbors - point_cloud_central, point_cloud_central], axis=-1)

    return edge_feature


def get_learning_rate(batch, base_lr, batch_size, decay_step, decay_rate, lr_clip):
    learning_rate = tf.train.exponential_decay(
        base_lr,             # Base learning rate.
        batch * batch_size,  # Current index into the dataset.
        decay_step,			 # Decay step.
        decay_rate,			 # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, lr_clip)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch, bn_init_decay, batch_size, bn_decay_step, bn_decay_rate, bn_decay_clip):
    bn_momentum = tf.train.exponential_decay(
        bn_init_decay,
        batch * batch_size,
        bn_decay_step,
        bn_decay_rate,
        staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay




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

