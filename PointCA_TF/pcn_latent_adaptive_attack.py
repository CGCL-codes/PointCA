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
import tensorflow_probability as tfp
import time
import datetime
import random
import h5py
from utils.data_util import create_dir,load_data, prepare_data_for_pcn_attack
from utils.pcn_tf_util import chamfer, chamfer_partial_inputs, earth_mover, pairwise_distance, knn, get_pc_neighbors, local_geometric_density, adaptive_project_constraint
from utils.visu_util import plot_pcd_three_views, plot_pcd_one_batch

#os.environ['CUDA_DEVICE_ORDER'] = '3,2,1,0' # cpu situation for debug

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=3, type=int)
parser.add_argument('--data_dir', default='data/eval_pcn_modelnet10')
parser.add_argument('--model_type', default='pcn_cd_layers')
parser.add_argument('--checkpoint_dir', default='ckpt/completion/pcn_cd')
parser.add_argument('--results_dir', default='results/attack_pcn_cd_modelnet10')
parser.add_argument('--num_input_points', type=int, default=1024)
parser.add_argument('--num_gt_points', type=int, default=16384)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--log_freq', type=int, default=40)
parser.add_argument('--num_iterations', type=int, default=200)
parser.add_argument("--num_pc_for_attack", type=int, default=20, help='Number of point clouds for attack (per shape class) [default: 20]')
parser.add_argument("--num_pc_for_target", type=int, default=5, help='Number of candidate point clouds for target (per point cloud for attack) [default: 5]')
parser.add_argument('--base_lr', type=float, default=0.5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--Lambda', type=float, default=1)
parser.add_argument('--knn', type=int, default=8, help='size of neighbor points set ')
parser.add_argument('--eta', type=float, default=1.25)
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
save_dir = os.path.join(output_path, 'latent_adaptive_attack_%s' % str(time_stamp))
os.makedirs(os.path.join(save_dir), exist_ok=True)
h5_dir = create_dir(os.path.join(save_dir, 'adv_example'))


# Output log file
log_file = os.path.join(save_dir, 'log_latent_adaptive_eta%.3f_Lambda%.3f.txt' % (args.eta, args.Lambda))
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
    inputs = tf.placeholder(tf.float32, (1, BATCH_SIZE * NUM_POINT, 3), 'inputs')
    inputs_adv = tf.placeholder(tf.float32, (1, BATCH_SIZE * NUM_POINT, 3), 'inputs_adv')
    npts = tf.placeholder(tf.int32, (BATCH_SIZE,), 'num_points')
    gt = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_GT_POINT, 3), 'ground_truths')
    pert_pl = tf.placeholder(tf.float32, (1, BATCH_SIZE * NUM_POINT, 3))
    #output = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))

    pc_parts = tf.squeeze(tf.split(inputs_adv, BATCH_SIZE, axis=1))
    adj_matrix = pairwise_distance(pc_parts)
    nn_idx = knn(adj_matrix, k=args.knn)
    point_cloud_neighbors = get_pc_neighbors(pc_parts, nn_idx=nn_idx)

    pert = tf.get_variable(name='pert', shape=[1, BATCH_SIZE * NUM_POINT, 3], initializer=tf.constant_initializer(), dtype=tf.float32)
    init_pert = tf.assign(pert, tf.truncated_normal([1, BATCH_SIZE * NUM_POINT, 3], mean=0, stddev=0.0001))
    load_pert = tf.assign(pert, pert_pl)
    #reset_pert = tf.assign(pert, tf.zeros([1, args.num_input_points, 3]))

    adv_pc = inputs_adv + pert

    model_module = importlib.import_module('.%s' % args.model_type, 'completion.models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0), is_training=False)
    model_adv = model_module.Model(adv_pc, npts, gt, tf.constant(1.0), is_training=False)

    latent_target_vector = model.features
    latent_adv_vector = model_adv.features
    loss_latent_L2 = tf.norm(latent_adv_vector - latent_target_vector, ord=2, axis=1)   # L2 loss
    # Args:  def __init__(self, dist_cls_a, dist_cls_b):
    #   dist_cls_a: the class of the first argument of the KL divergence.
    #   dist_cls_b: the class of the second argument of the KL divergence.
    P = tf.distributions.Categorical(probs=tf.nn.softmax(latent_target_vector))
    Q = tf.distributions.Categorical(probs=tf.nn.softmax(latent_adv_vector))
    loss_latent_kl = tfp.distributions.kl_divergence(P, Q)   # target adv

    output_adv = model_adv.outputs
    cd_op, cd_per_pc = chamfer(output_adv, gt)
    emd_op = earth_mover(output_adv, gt)

    loss_dist, dist_per_pc = chamfer_partial_inputs(inputs_adv, adv_pc, BATCH_SIZE)
    loss = tf.reduce_mean(loss_latent_L2) + args.Lambda*tf.reduce_mean(loss_latent_kl)

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
    # pc_idx_file_path = os.path.join(save_dir, 'attack_pc_ids.npy')
    # np.save(pc_idx_file_path, attack_pc_idx)


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
            npts_batch = np.stack([x.shape[0] if x.shape[0] < NUM_POINT else NUM_POINT for x in p_source_pc_batch]).astype(np.int32)
            p_source_pc = np.expand_dims(np.concatenate([x for x in p_source_pc_batch]), 0).astype(np.float32)
            p_target_pc = np.expand_dims(np.concatenate([x for x in p_target_pc_batch]), 0).astype(np.float32)

            source_gt_batch = source_gt[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            target_pc_batch = target_gt[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            source_batch_labels = source_labels[BATCH_SIZE*b: BATCH_SIZE*(b+1)]
            target_batch_labels = target_labels[BATCH_SIZE*b: BATCH_SIZE*(b+1)]


            # Verify the original completion performance
            feed_dicts = {inputs_adv: p_target_pc, npts: npts_batch, gt: target_pc_batch}
            #chamfer_distance = sess.run(cd_op, feed_dict=feed_dicts)
            #print('verify the completion performance cd: %f '%chamfer_distance)
            feed_dicts[pert_pl] = np.zeros_like(p_target_pc, dtype=np.float32)
            _ = sess.run(load_pert, feed_dict=feed_dicts)
            completion = sess.run(output_adv, feed_dict=feed_dicts) # original completion result
            completion_cd_per_pc = sess.run(cd_per_pc, feed_dict=feed_dicts)

            # Calculate density map
            feed_dicts = {inputs_adv: p_source_pc, npts: npts_batch, gt: source_gt_batch}
            #parts = sess.run(pc_parts, feed_dict=feed_dicts)
            pc_n= sess.run(point_cloud_neighbors, feed_dict=feed_dicts)
            density_map = local_geometric_density(pc_n, p_source_pc_batch)
            density_map = np.expand_dims(np.concatenate([x for x in density_map]), 0)


            _ = sess.run(init_pert)
            for iteration in range(args.num_iterations):    # args.num_iterations
                feed_dicts = {inputs_adv: p_source_pc, inputs: p_target_pc, npts: npts_batch, gt: target_pc_batch}
                _ = sess.run(attack_op, feed_dict=feed_dicts)
                adv_point_cloud = sess.run(adv_pc, feed_dict=feed_dicts)

                adv_pc_update = adaptive_project_constraint(adv_point_cloud, density_map, args.eta, p_source_pc)
                delta = adv_pc_update - p_source_pc

                feed_dicts[pert_pl] = delta
                _ = sess.run(load_pert, feed_dict=feed_dicts)

                if iteration % args.log_freq == 0 or iteration == args.num_iterations - 1:
                    log_string('------------- STEP: {} -------------'.format(iteration))
                    losses, output_chamfer, input_chamfer = sess.run([loss, cd_op, loss_dist], feed_dict=feed_dicts)
                    log_string('loss: %.8f   cd_output: %.8f   pert_dist: %.8f' %(losses, output_chamfer, input_chamfer))

            #Evaluate one batch
            cd_outputs, cd_per_output, emd = sess.run([cd_op, cd_per_pc, emd_op], feed_dict=feed_dicts)   # evaluating the outpus
            cd_inputs, cd_per_input = sess.run([loss_dist, dist_per_pc], feed_dict=feed_dicts)    # evaluating the inputs
            adv_pc_inputs, adv_pc_outputs = sess.run([adv_pc, output_adv], feed_dict=feed_dicts) # the adv_examples and reconstruction results
            adv_pc_inputs = np.squeeze(np.split(adv_pc_inputs, BATCH_SIZE, axis=1))
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
    df.to_csv(os.path.join(save_dir, 'statistic_table_latent_adaptive_eta%s_lambda%s.csv' % (str(args.eta), str(args.Lambda))), encoding="utf-8-sig", mode="a", header=True, index=False)

    log_string('Total time %s' % str(datetime.timedelta(seconds=time.time() - start)))
    sess.close()



if __name__ == '__main__':

    attack(args)
