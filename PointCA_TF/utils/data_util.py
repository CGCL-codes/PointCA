import  os
import numpy as np, tensorflow as tf
import  pdb
import  h5py
from tensorpack import dataflow


def resample_pcd(pcd, n):
    """drop or duplicate points so that input of each object has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt

class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        """get the number of batches"""
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)  # int(False) == 0

    def __iter__(self):
        """generating data in batches"""
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]  # reset holder as empty list => holder = []
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        """
        Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        """
        ids = np.stack([x[0] for x in data_holder])
        #inputs = [x[1] for x in data_holder]
        inputs = [resample_pcd(x[1], self.input_size) if x[1].shape[0] > self.input_size else x[1] for x in data_holder]
        inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(np.float32)
        #npts = np.stack([x[1].shape[0]  for x in data_holder]).astype(np.int32)
        npts = np.stack([x[1].shape[0] if x[1].shape[0] < self.input_size else self.input_size
            for x in data_holder]).astype(np.int32)
        #gts = np.stack([x[2] for x in data_holder]).astype(np.float32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        return ids, inputs, npts, gts

def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
    """load LMDB files, then generateing batches??"""
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)  # buffer_size
        df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)  # multiprocess the data
    df = BatchData(df, batch_size, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size

def get_queued_data(generator, dtypes, shapes, queue_capacity=10):
    assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'
    queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)
    placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=True)
    feed_fn = lambda: {placeholder: value for placeholder, value in zip(placeholders, next(generator))}
    queue_runner = tf.contrib.training.FeedingQueueRunner(
        queue, [enqueue_op], close_op, feed_fns=[feed_fn])
    tf.train.add_queue_runner(queue_runner)
    return queue.dequeue()




''' === build for PointCA === '''

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def load_data(data_path, file_list, base_name_list):
    data_list = [None] * len(base_name_list)

    for i, base_name in enumerate(base_name_list):
        file_name = [f for f in file_list if base_name in f][0]
        data_list[i] = np.load(os.path.join(data_path, file_name))

    if len(data_list) == 1:
        data_list = data_list[0]

    return data_list


def prepare_data_for_pcn_attack(pc_classes, source_classes_for_attack, classes_p_data, classes_c_data, slice_idx, attack_pc_idx, num_pc_for_target, nn_idx_mat, labels):
    num_classes = len(pc_classes)

    source_data_list = []
    source_c_data_list = []
    target_data_list = []
    target_p_data_list = []
    source_names_list = []
    target_names_list = []

    for i in range(num_classes):
        source_class_name = pc_classes[i]
        if source_class_name not in source_classes_for_attack:
            continue

        source_attack_idx = attack_pc_idx[i]
        num_source_pc_for_attack = len(source_attack_idx)

        source_class_data = classes_p_data[slice_idx[i]:slice_idx[i + 1]]
        source_class_data_for_attack = source_class_data[source_attack_idx]
        source_class_c_data = classes_c_data[slice_idx[i]:slice_idx[i + 1]]
        source_class_c_data_for_attack = source_class_c_data[source_attack_idx]
        source_class_names = labels[slice_idx[i]:slice_idx[i + 1]]
        source_class_names_for_attack = source_class_names[source_attack_idx]


        num_attack_per_pc = 0
        target_data_for_attack_list = []
        target_p_data_for_attack_list = []
        target_names_for_attack_list = []

        for j in range(num_classes):
            target_class_name = pc_classes[j]
            if target_class_name not in pc_classes or target_class_name == source_class_name:
                continue

            nn_idx_s_class_t_class = nn_idx_mat[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]]
            nn_idx_s_for_attack_t_class = nn_idx_s_class_t_class[source_attack_idx].copy()


            num_attack_per_pc += num_pc_for_target

            target_class_data = classes_c_data[slice_idx[j]:slice_idx[j + 1]]
            target_class_p_data = classes_p_data[slice_idx[j]:slice_idx[j + 1]]
            target_class_names = labels[slice_idx[j]:slice_idx[j + 1]]
            target_class_data_for_attack_list = []
            target_class_p_data_for_attack_list = []
            target_class_names_for_attack_list = []

            for s in range(num_source_pc_for_attack):
                target_attack_idx = nn_idx_s_for_attack_t_class[s, :num_pc_for_target]
                target_class_data_for_attack_curr = target_class_data[target_attack_idx]
                target_class_data_for_attack_list.append(np.expand_dims(target_class_data_for_attack_curr, axis=0))
                target_class_p_data_for_attack_curr = target_class_p_data[target_attack_idx]
                target_class_p_data_for_attack_list.append(np.expand_dims(target_class_p_data_for_attack_curr, axis=0))
                target_class_names_for_attack_curr = target_class_names[target_attack_idx]
                target_class_names_for_attack_list.append(target_class_names_for_attack_curr)

            target_data_for_attack = np.vstack(target_class_data_for_attack_list)
            target_data_for_attack_list.append(target_data_for_attack)
            target_p_data_for_attack = np.vstack(target_class_p_data_for_attack_list)
            target_p_data_for_attack_list.append(target_p_data_for_attack)
            target_names_for_attack = np.vstack(target_class_names_for_attack_list)
            target_names_for_attack_list.append(target_names_for_attack)

        target_data_for_attack_concat = np.concatenate(target_data_for_attack_list, axis=1)
        target_p_data_for_attack_concat = np.concatenate(target_p_data_for_attack_list, axis=1)
        target_names_for_attack_concat = np.concatenate(target_names_for_attack_list,axis=1)
        old_shape = target_data_for_attack_concat.shape
        new_shape = [old_shape[0]*old_shape[1]] + [old_shape[n] for n in range(2, len(old_shape))]
        target_data_curr = np.reshape(target_data_for_attack_concat, new_shape)
        old_shape = target_p_data_for_attack_concat.shape
        new_shape = [old_shape[0]*old_shape[1]] + [old_shape[n] for n in range(2, len(old_shape))]
        target_p_data_curr = np.reshape(target_p_data_for_attack_concat, new_shape)
        old_shape = target_names_for_attack_concat.shape
        new_shape = [old_shape[0] * old_shape[1]]
        target_names_curr = np.reshape(target_names_for_attack_concat, new_shape)


        target_data_list.append(target_data_curr)
        target_p_data_list.append(target_p_data_curr)
        #target_names_list.append(target_names_curr)

        source_data_curr = np.vstack([[source_class_data_for_attack[n]] * num_attack_per_pc for n in range(num_source_pc_for_attack)])
        source_data_list.append(source_data_curr)
        source_c_data_curr = np.vstack([[source_class_c_data_for_attack[n]] * num_attack_per_pc for n in range(num_source_pc_for_attack)])
        source_c_data_list.append(source_c_data_curr)
        source_names_curr = np.vstack([[source_class_names_for_attack[n]] * num_attack_per_pc for n in range(num_source_pc_for_attack)])
        source_names_curr = np.reshape(source_names_curr, new_shape)
        #source_names_list.append(source_names_curr)

    source_data = np.vstack(source_data_list)
    source_c_data = np.vstack(source_c_data_list)
    #source_names = np.vstack(source_names_list)
    target_data = np.vstack(target_data_list)
    target_p_data = np.vstack(target_p_data_list)
    #target_names = np.vstack(target_names_list)

    return source_data, source_c_data, source_names_curr, target_p_data, target_data, target_names_curr


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    # label = f['label'][:]
    return data


def load_adv_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    #print('f.keys() :', list(f.keys()))
    source = f['source'][:]
    adv = f['adv'][:]
    return (source, adv)


def load_h5_cls_adv(h5_filename):
    f = h5py.File(h5_filename, 'r')
    print('f.keys() :', list(f.keys()))
    clsadv = f['clsadv'][:]

    return  clsadv


def load_h5_analysis(h5_filename):
    f = h5py.File(h5_filename, 'r')
    #print('f.keys() :', list(f.keys()))
    source = f['source'][:]
    #adv = f['adv'][:]
    #target = f['target'][:]
    groundtruth = f['gt'][:]
    return (source, groundtruth)


def load_h5_adv_analysis(h5_filename):
    f = h5py.File(h5_filename, 'r')
    #print('f.keys() :', list(f.keys()))
    source = f['source'][:]
    adv = f['adv'][:]
    target = f['target'][:]
    groundtruth = f['gt'][:]
    return (source, adv, target, groundtruth)


def load_h5_visual(h5_filename):
    f = h5py.File(h5_filename, 'r')
    #print('f.keys() :', list(f.keys()))
    source = f['source'][:]
    adv = f['adv'][:]
    output = f['output'][:]
    completion = f['completion'][:]
    target = f['target'][:]
    gt = f['gt'][:]
    return source, adv, output, completion, target, gt