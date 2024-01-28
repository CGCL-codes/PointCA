import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import argparse
import pandas
import importlib
import numpy as np
import tensorflow as tf
import time
import datetime
import random
import h5py
from utils.data_util import create_dir,load_data, prepare_data_for_pcn_attack
from utils.rfa_tf_util import chamfer_per_pc, earth_mover, channelwise_project_constraint
from utils.visu_util import plot_pcd_three_views, plot_pcd_one_batch

#os.environ['CUDA_DEVICE_ORDER'] = '3,2,1,0' # cpu situation for debug

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=3, type=int)
parser.add_argument('--data_dir', default='data/eval_pcn_modelnet10')
parser.add_argument('--model_type', default='rfa')
parser.add_argument('--checkpoint_dir', default='ckpt/completion/rfa')
parser.add_argument('--results_dir', default='results/attack_rfa_modelnet10')
parser.add_argument('--num_input_points', type=int, default=1024)
parser.add_argument('--num_gt_points', type=int, default=16384)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--log_freq', type=int, default=40)
parser.add_argument('--num_iterations', type=int, default=200)
parser.add_argument("--num_pc_for_attack", type=int, default=20, help='Number of point clouds for attack (per shape class) [default: 20]')
parser.add_argument("--num_pc_for_target", type=int, default=5, help='Number of candidate point clouds for target (per point cloud for attack) [default: 5]')
parser.add_argument('--base_lr', type=float, default=0.5)
parser.add_argument('--decay_rate', type=float, default=0.6)
parser.add_argument('--epsilon', type=float, default=0.06)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


# Default parameters, CANNOT be modified
BATCH_SIZE = args.batch_size
NUM_POINT = args.num_input_points
NUM_GT_POINT = args.num_gt_points
NUM_FOR_ATTACK = args.num_pc_for_attack
NUM_FOR_TARGET = args.num_pc_for_target


time_stamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
output_path = create_dir(args.results_dir)
save_dir = os.path.join(output_path, 'geometry_channelwise_attack_%s' % str(time_stamp))
os.makedirs(os.path.join(save_dir), exist_ok=True)
h5_dir = create_dir(os.path.join(save_dir, 'adv_example'))


# Output log file
log_file = os.path.join(save_dir, 'log_geometry_channelwise_epsilon%.3f.txt' % args.epsilon)
log_fout = open(log_file, 'w+')

def log_string(out_str):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)

for arg in sorted(vars(args)):
    log_string(arg + ': ' + str(getattr(args, arg)))


# Evaluate the robustness of the completion model
def attack(args):

    # Build the computation graph
    inputs = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINT, 3), 'inputs')
    gt = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_GT_POINT, 3), 'ground_truths')
    pert_pl = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINT, 3))
    #output = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))

    pert = tf.get_variable(name='pert', shape=[ BATCH_SIZE, NUM_POINT, 3], initializer=tf.constant_initializer(), dtype=tf.float32)
    init_pert = tf.assign(pert, tf.truncated_normal([BATCH_SIZE, NUM_POINT, 3], mean=0, stddev=0.0001))
    load_pert = tf.assign(pert, pert_pl)
    #reset_pert = tf.assign(pert, tf.zeros([1, args.num_input_points, 3]))

    adv_pc = inputs + pert

    model_module = importlib.import_module('.%s' % args.model_type, 'model.completion')
    model = model_module.Model(adv_pc, gt, tf.constant(1.0), is_training=False)

    output = model.outputs
    cd_op, cd_per_pc = chamfer_per_pc(output, gt)
    emd_op = earth_mover(output, gt)

    loss_dist, dist_per_pc = chamfer_per_pc(inputs, adv_pc)
    loss = cd_op

    lr = tf.train.exponential_decay(learning_rate=args.base_lr, global_step=args.num_iterations, decay_steps=20, decay_rate=args.decay_rate, staircase=True, name="learning_rate_decay")
    lr = tf.maximum(lr, 2e-4)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    attack_op = optimizer.minimize(loss, var_list=pert)

    vl = tf.global_variables()
    vl = [x for x in vl if 'pert' not in x.name]
    saver = tf.train.Saver(vl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)


    # attack 
    start = time.time()

    total_cd = 0 # output & target
    total_emd = 0
    total_dist = 0 # adv_pc & original partial pc

    files = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
    partial_point_clouds, complete_point_clouds, pc_classes, slice_idx, names_list = \
        load_data(args.data_dir, files,
                  ['partial_pc_set', 'complete_pc_set', 'pc_classes', 'slice_idx_set', 'names_list'])
    nn_idx = load_data(args.data_dir, files, ['chamfer_nn_idx_complete_set'])

    attack_pc_idx = np.zeros(shape=(len(pc_classes), NUM_FOR_ATTACK), dtype=np.int32)
    class_cd_statistic = np.zeros(shape=(len(pc_classes), len(pc_classes)-1), dtype=np.float32)
    class_emd_statistic = np.zeros(shape=(len(pc_classes), len(pc_classes)-1), dtype=np.float32)
    class_dist_statistic = np.zeros(shape=(len(pc_classes), len(pc_classes) - 1), dtype=np.float32)
    source_labels_statistic = np.array([])
    target_labels_statistic = np.array([])
    t_nre_statistic = np.array([])
    cd_statistic_per_pc = np.array([])
    completion_statistic_per_pc = np.array([])
    pert_dist_statistic_per_pc = np.array([])


    for k in range(len(pc_classes)):
        end_id = slice_idx[k + 1] - slice_idx[k]
        random.seed(k)
        attack_pc_idx[k] = random.sample(range(0, end_id), NUM_FOR_ATTACK)
    #pc_idx_file_path = os.path.join(save_dir, 'attack_pc_ids.npy')
    #np.save(pc_idx_file_path, attack_pc_idx)


    for c in range(len(pc_classes)):    # attack for each class     len(pc_classes)
        class_cd = 0
        class_emd = 0
        class_dist = 0
        pc_class_name = pc_classes[c]
        log_string('############################# CLASS: {} #############################'.format(pc_class_name))
        plots_dir = create_dir(os.path.join(save_dir, pc_class_name))
        source_partial, source_gt, source_labels, target_partial, target_gt, target_labels = prepare_data_for_pcn_attack(
            pc_classes, [pc_class_name], partial_point_clouds, complete_point_clouds, slice_idx,attack_pc_idx, NUM_FOR_TARGET, nn_idx, names_list)

        assert len(source_labels) % BATCH_SIZE == 0, 'The number of examples (%d) should be divided by the batch size (%d)' % (len(source_labels),BATCH_SIZE)
        n_batches = len(source_labels) // BATCH_SIZE

        for b in range(n_batches):   # generating adv_examples by # of batchs    # n_batches
            log_string('**************************** BATCH: {} ****************************'.format(b))
            p_source_pc_batch = source_partial[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            p_target_pc_batch = target_partial[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            p_source_pc = np.copy(p_source_pc_batch)
            p_target_pc = np.copy(p_target_pc_batch)
            source_gt_batch = source_gt[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            target_pc_batch = target_gt[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            source_batch_labels = source_labels[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            target_batch_labels = target_labels[BATCH_SIZE*b: BATCH_SIZE*(b+1)]


            # Verify the original completion performance
            feed_dicts = {inputs: p_target_pc, gt: target_pc_batch}
            #chamfer_distance = sess.run(cd_op, feed_dict=feed_dicts)
            #print('verify the completion performance cd: %f '%chamfer_distance)
            feed_dicts[pert_pl] = np.zeros_like(p_target_pc, dtype=np.float32)
            _ = sess.run(load_pert, feed_dict=feed_dicts)
            completion = sess.run(output, feed_dict=feed_dicts) # original completion result
            completion_cd_per_pc = sess.run(cd_per_pc, feed_dict=feed_dicts)


            sess.run(init_pert)
            for iteration in range(args.num_iterations):    # args.num_iterations
                feed_dicts = {inputs: p_source_pc, gt: target_pc_batch}
                _ = sess.run(attack_op, feed_dict=feed_dicts)
                adv_point_cloud = sess.run(adv_pc, feed_dict=feed_dicts)

                delta = channelwise_project_constraint(adv_point_cloud, -args.epsilon, args.epsilon, p_source_pc)

                feed_dicts[pert_pl] = delta
                _ = sess.run(load_pert, feed_dict=feed_dicts)

                if iteration % args.log_freq == 0 or iteration == args.num_iterations - 1:
                    log_string('------------- STEP: {} -------------'.format(iteration))
                    losses, output_chamfer, input_chamfer = sess.run([loss, cd_op, loss_dist], feed_dict=feed_dicts)
                    log_string('loss: %.8f   adv_loss: %.8f   pert_dist: %.8f' %(losses, output_chamfer, input_chamfer))

            #Evaluate one batch
            cd_outputs, cd_per_output, emd = sess.run([cd_op, cd_per_pc, emd_op], feed_dict=feed_dicts)   # evaluating the outpus
            cd_inputs, cd_per_input = sess.run([loss_dist, dist_per_pc], feed_dict=feed_dicts)    # evaluating the inputs
            adv_pc_inputs, adv_pc_outputs = sess.run([adv_pc, output], feed_dict=feed_dicts) # the adv_examples and reconstruction results
            file_name = 'Source_%s_%s_Target_%s' % (source_batch_labels[0].split('_')[0], source_batch_labels[0].split('_')[-2], target_batch_labels[0].split('_')[0])
            plot_path = os.path.join(plots_dir, file_name + '.png')   # names depending on batchsize
            plot_pcd_one_batch(plot_path, [p_source_pc_batch, adv_pc_inputs, adv_pc_outputs, completion, target_pc_batch, source_gt_batch],
                               ['source', 'adv', 'output', 'completion', 'target', 'ground truth'])

            # Save the pcds
            examples_data = h5py.File(os.path.join(h5_dir, file_name+'.h5'), "w")
            examples_data.create_dataset('source', data=p_source_pc_batch)
            examples_data.create_dataset('adv', data=adv_pc_inputs)
            examples_data.create_dataset('output', data=adv_pc_outputs)
            examples_data.create_dataset('completion', data=completion)
            examples_data.create_dataset('target', data=target_pc_batch)
            examples_data.create_dataset('gt', data=source_gt_batch)
            examples_data.close()

            # Collect statistic results
            source_labels_statistic = np.concatenate((source_labels_statistic, source_batch_labels), axis=0)
            target_labels_statistic = np.concatenate((target_labels_statistic, target_batch_labels), axis=0)
            cd_statistic_per_pc = np.concatenate((cd_statistic_per_pc, cd_per_output), axis=0)
            completion_statistic_per_pc = np.concatenate((completion_statistic_per_pc, completion_cd_per_pc), axis=0)
            t_nre = cd_per_output / completion_cd_per_pc
            t_nre_statistic = np.concatenate((t_nre_statistic, t_nre), axis=0)
            pert_dist_statistic_per_pc = np.concatenate((pert_dist_statistic_per_pc, cd_per_input), axis=0)
            class_cd_statistic[c, b % (len(pc_classes) - 1)] += cd_outputs # each target class for 20 times
            class_emd_statistic[c, b % (len(pc_classes) - 1)]+= emd
            class_dist_statistic[c, b % (len(pc_classes) - 1)]+= cd_inputs
            class_cd += cd_outputs * BATCH_SIZE     # calculating one batch
            class_emd += emd * BATCH_SIZE
            class_dist += cd_inputs *BATCH_SIZE


        total_cd += class_cd / len(source_labels)      # class average
        total_emd += class_emd / len(source_labels)
        total_dist += class_dist / len(source_labels)

    total_cd = total_cd / len(pc_classes)
    total_emd = total_emd / len(pc_classes)
    total_dist = total_dist / len(pc_classes)
    class_cd_statistic= class_cd_statistic / NUM_FOR_ATTACK
    class_emd_statistic = class_emd_statistic / NUM_FOR_ATTACK
    class_dist_statistic = class_dist_statistic / NUM_FOR_ATTACK

    # log and save
    log_string('the total mean cd : %f' % total_cd)
    log_string('the total mean emd : %f' % total_emd)
    log_string('the total mean dist : %f' % total_dist)
    log_string('the total mean t-nre : %f' % np.mean(t_nre_statistic))
    class_cd_statistic_file_path = os.path.join(save_dir, 'class_cd_statistic.npy')
    np.save(class_cd_statistic_file_path, class_cd_statistic)
    class_emd_statistic_file_path = os.path.join(save_dir, 'class_emd_statistic.npy')
    np.save(class_emd_statistic_file_path, class_emd_statistic)
    class_dist_statistic_file_path = os.path.join(save_dir, 'class_dist_statistic.npy')
    np.save(class_dist_statistic_file_path, class_dist_statistic)
    data_csv = {"source_id": source_labels_statistic, "target_id": target_labels_statistic, "dist": pert_dist_statistic_per_pc, "completion_re": completion_statistic_per_pc, "outputs_re": cd_statistic_per_pc, "T-NRE": t_nre_statistic}
    df = pandas.DataFrame(data_csv)
    df.to_csv(os.path.join(save_dir, 'statistic_table_geometry_channelwise_epsilon%.3f.csv' %args.epsilon), encoding="utf-8-sig", mode="a", header=True, index=False)

    log_string('Total time %s' % str(datetime.timedelta(seconds=time.time() - start)))
    sess.close()



if __name__ == '__main__':

    attack(args)
