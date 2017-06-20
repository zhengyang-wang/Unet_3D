import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H53DDataLoader
# from utils.img_utils import imsave
from utils import ops
from utils.Dense_Transformer_Networks_3D import *


"""
This module build a standard U-NET for semantic segmentation.
If want VAE using pixelDCL, please visit this code:
https://github.com/HongyangGao/UVAE
"""


class Unet_3D(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.savedir):
            os.makedirs(conf.savedir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        if self.conf.data_type == '3D':
            self.conv_size = (3, 3, 3)
            self.pool_size = (2, 2, 2)
            self.axis, self.channel_axis = (1, 2, 3), 4
            self.input_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width]
            if self.conf.add_dtn == True:
                # Operation in the list is trying to get the input shape of Dense transformer networks.
                self.dtn_input_shape = [self.conf.batch,int(self.conf.depth/(2**self.conf.dtn_location)),\
                                        int(self.conf.height/(2**self.conf.dtn_location)),
                                        int(self.conf.width/(2**self.conf.dtn_location)),
                                        self.conf.start_channel_num*(2**self.conf.dtn_location)]
                self.transform = DSN_Transformer_3D(self.dtn_input_shape,self.conf.control_points_ratio)
                self.insertdtn = self.conf.dtn_location
            else:
                self.insertdtn = -1

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(
            tf.int64, self.output_shape, name='annotations')
        self.predictions = self.inference(self.inputs)
        self.cal_loss()

    def cal_loss(self):
        expand_annotations = tf.expand_dims(
            self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(
            expand_annotations, axis=[self.channel_axis],
            name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(
            one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        losses = tf.losses.softmax_cross_entropy(
            one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        self.decoded_predictions = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.annotations, self.decoded_predictions,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        self.softmax_predictions = tf.nn.softmax(self.predictions)

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            if layer_index == self.insertdtn:
                outputs = self.build_down_block(outputs, name, down_outputs,first=is_first,TPS = True)
            else:
                outputs = self.build_down_block(outputs, name, down_outputs, first=is_first,TPS = False)  
            print("down ",layer_index," shape ", outputs.get_shape())          
        outputs = self.build_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            if layer_index == self.insertdtn:
                outputs = self.build_up_block(outputs, down_inputs, name,final=is_final,Decoder=True )
            else:
                outputs = self.build_up_block(outputs, down_inputs, name,final=is_final,Decoder=False )
            print("up ",layer_index," shape ",outputs.get_shape())
        return outputs

    def build_down_block(self, inputs, name, down_outputs, first=False,TPS=False):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = ops.conv(inputs, out_num, self.conv_size,
                         name+'/conv1', self.conf.data_type)
        if TPS == True:
            conv1= self.transform.Encoder(conv1,conv1)
        conv2 = ops.conv(conv1, out_num, self.conv_size,
                         name+'/conv2', self.conf.data_type)
        down_outputs.append(conv2)
        pool = ops.pool(conv2, self.pool_size, name +
                        '/pool', self.conf.data_type)
        return pool

    def build_bottom_block(self, inputs, name):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = ops.conv(
            inputs, 2*out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv2 = ops.conv(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        return conv2

    def build_up_block(self, inputs, down_inputs, name, final=False,Decoder=False):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.conv_func()(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        if Decoder == True:
            conv2 = self.transform.Decoder(conv2,conv2)
        out_num = self.conf.class_num if final else out_num/2
        conv3 = ops.conv(
            conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        data_reader = H53DDataLoader(self.conf.data_dir, self.conf.patch_size, self.conf.validation_id, self.conf.overlap_stepsize, self.conf.aug_flip, self.conf.aug_rotate)
        for train_step in range(1, self.conf.max_step+1):
            # if train_step % self.conf.test_interval == 0:
            #     inputs, annotations = data_reader.valid_next_batch()
            #     feed_dict = {self.inputs: inputs,
            #                  self.annotations: annotations}
            #     loss, summary = self.sess.run(
            #         [self.loss_op, self.valid_summary], feed_dict=feed_dict)
            #     self.save_summary(summary, train_step+self.conf.reload_step)
            #     print('----testing loss', loss)
            # el
            if train_step % self.conf.summary_interval == 0:
                inputs, annotations = data_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, train_step+self.conf.reload_step)
            else:
                inputs, annotations = data_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                print('----training loss', loss)
            if train_step % self.conf.save_interval == 0:
                self.save(train_step+self.conf.reload_step)

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        data_reader = H53DDataLoader(self.conf.data_dir, self.conf.patch_size, self.conf.validation_id, self.conf.overlap_stepsize)
        self.sess.run(tf.local_variables_initializer())
        # count = 0
        losses = []
        accuracies = []
        for i in range(data_reader.num_of_valid_patches):
            inputs, annotations, _ = data_reader.valid_next_batch()
            # if inputs.shape[0] < self.conf.batch:
            #     break

            # pseudo_inputs = np.zeros((1,32,32,32,2), dtype=np.float32)
            # pseudo_labels = np.zeros((1,32,32,32), dtype=np.float32)
            # CUT_MEAN = np.array((100.913811861, 121.187003401), dtype=np.float32)
            # pseudo_inputs -= CUT_MEAN

            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            loss, accuracy = self.sess.run([self.loss_op, self.accuracy_op],feed_dict=feed_dict)
            print('values----->', loss, accuracy)
            # count += 1
            losses.append(loss)
            accuracies.append(accuracy)
        print('Loss: ', np.mean(losses))
        print('Accuracy: ', np.mean(accuracies))

    def predict(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        data_reader = H53DDataLoader(self.conf.data_dir, self.conf.patch_size, self.conf.validation_id, self.conf.overlap_stepsize)
        self.sess.run(tf.local_variables_initializer())
        predictions = {}
        for i in range(data_reader.num_of_valid_patches):
            inputs, annotations, location = data_reader.valid_next_batch()
            # if inputs.shape[0] < self.conf.batch:
            #     break
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            preds = self.sess.run(self.softmax_predictions, feed_dict=feed_dict)
            print('--->processing results for: ', location)
            for j in range(self.conf.patch_size):
                for k in range(self.conf.patch_size):
                    for l in range(self.conf.patch_size):
                        key = (location[0]+j, location[1]+k, location[2]+l)
                        if key not in predictions.keys():
                            predictions[key] = []
                        predictions[key].append(preds[0, j, k, l, :])
        print('--->averaging results')
        results = np.zeros((data_reader.t_d, data_reader.t_h, data_reader.t_w, self.conf.class_num), dtype=np.float32)
        for key in predictions.keys():
            results[key[0], key[1], key[2]] = np.mean(predictions[key], axis=0)
        print('--->saving results')
        save_filename = 'results' + str(self.conf.test_step) + '_sub' + str(self.conf.validation_id) + '_overlap' + str(self.conf.overlap_stepsize) +'.npy'
        save_file = os.path.join(self.conf.savedir, save_filename)
        np.save(save_file, results)

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
