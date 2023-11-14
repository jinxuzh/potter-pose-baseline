# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
import torch

class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    all_input -- list of input image tensors ~ (N_take, F, 3, 224, 224)
    poses_3d -- list of ground-truth 3D poses ~ (N_take, F, 21, 3)
    all_vis_flag -- list of visible flags ~ (N_take, F, 21)
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, all_input, poses_3d, all_vis_flag,
                 chunk_length, pad=0, causal_shift=0, shuffle=True, random_seed=1234, seq_to_one=True,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None, endless=False):
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_3d)):
            n_chunks = (poses_3d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_3d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)

        self.input_img_tensor = all_input
        self.poses_3d = poses_3d
        self.all_vis_flag = all_vis_flag
        
        # Initialize buffers
        self.seq_to_one = seq_to_one
        if self.seq_to_one:
            self.batch_input = torch.empty((batch_size, chunk_length + 2*pad, *all_input[0].shape[1:]))
            self.batch_3d = torch.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
            self.batch_vis_flag = torch.empty((batch_size, chunk_length, all_vis_flag[0].shape[-1]))
        else:
            self.batch_input = torch.empty((batch_size, chunk_length + 2*pad, *all_input[0].shape[1:]))
            self.batch_3d = torch.empty((batch_size, chunk_length + 2*pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
            self.batch_vis_flag = torch.empty((batch_size, chunk_length + 2*pad, all_vis_flag[0].shape[-1]))
            raise Exception('Loss for seq-to-seq hasn\'t been implemented yet.')

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        self.poses_3d = poses_3d
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs() # paris: (N_total, 4)
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_frame, end_frame, flip) in enumerate(chunks):
                    # Start and end frame index of middle frame
                    start = start_frame - self.pad - self.causal_shift
                    end = end_frame + self.pad - self.causal_shift
                    # Get current chunk from sequence
                    seq_img_tensor = self.input_img_tensor[seq_i] # (F, 3, 224, 224)
                    seq_3d = self.poses_3d[seq_i]                 # (F, 21, 3)
                    seq_vis_flag = self.all_vis_flag[seq_i]       # (F, 21)
                    # Compute valid start and end to pad
                    start_valid = max(start, 0)
                    end_valid = min(end, seq_img_tensor.shape[0])
                    pad_left = start_valid - start
                    pad_right = end - end_valid
                    # If sequence-to-one, only return middle frame's pose3d as target
                    if self.seq_to_one:
                        self.batch_input[i] = torch.from_numpy(np.pad(seq_img_tensor[start_valid:end_valid].numpy(), 
                                                                      ((pad_left, pad_right), (0, 0), (0, 0), (0,0)), 'edge'))
                        self.batch_3d[i] = seq_3d[start_frame:end_frame]
                        self.batch_vis_flag[i] = seq_vis_flag[start_frame:end_frame]
                    else:
                        self.batch_input[i] = torch.from_numpy(np.pad(seq_img_tensor[start_valid:end_valid].numpy(), 
                                                                      ((pad_left, pad_right), (0, 0), (0, 0), (0,0)), 'edge'))
                        self.batch_3d[i] = torch.from_numpy(np.pad(seq_3d[start_valid:end_valid].numpy(), 
                                                                   ((pad_left, pad_right), (0, 0), (0, 0)), 'edge'))
                        self.batch_vis_flag[i] = torch.from_numpy(np.pad(seq_vis_flag[start_valid:end_valid].numpy(),
                                                                         ((pad_left, pad_right), (0, 0)), 'edge'))
                    # TODO: Add data augmentation
                    if flip:
                        pass

                yield self.batch_input[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_vis_flag[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False
            

class testChunkedGenerator:
    """
    Batched data generator used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    Arguments:
    batch_size -- the batch size to use for training
    all_input -- list of input image tensors ~ (N_take, F, 3, 224, 224)
    poses_3d -- list of ground-truth 3D poses ~ (N_take, F, 21, 3)
    all_vis_flag -- list of visible flags ~ (N_take, F, 21)
    all_hand_wrist -- list of hand wrist 3D kpts ~ (N_take, F, 3)
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, batch_size, all_input, poses_3d, all_vis_flag, all_hand_wrist,
                 chunk_length, pad=0, causal_shift=0, shuffle=True, random_seed=1234, seq_to_one=True):
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_3d)):
            n_chunks = (poses_3d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_3d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)

        self.input_img_tensor = all_input
        self.poses_3d = poses_3d
        self.all_vis_flag = all_vis_flag
        self.all_hand_wrist = all_hand_wrist
        
        # Initialize buffers
        self.seq_to_one = seq_to_one
        if self.seq_to_one:
            self.batch_input = torch.empty((batch_size, chunk_length + 2*pad, *all_input[0].shape[1:])) # (B,F,3,224,224)
            self.batch_3d = torch.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1])) # (B,1,21,3)
            self.batch_vis_flag = torch.empty((batch_size, chunk_length, all_vis_flag[0].shape[-1])) # (B,1,21)
            self.batch_hand_wrist = torch.empty((batch_size, chunk_length, all_hand_wrist[0].shape[-1])) # (B,1,21)
        else:
            self.batch_input = torch.empty((batch_size, chunk_length + 2*pad, *all_input[0].shape[1:]))
            self.batch_3d = torch.empty((batch_size, chunk_length + 2*pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
            self.batch_vis_flag = torch.empty((batch_size, chunk_length + 2*pad, all_vis_flag[0].shape[-1]))
            self.batch_hand_wrist = torch.empty((batch_size, chunk_length + 2*pad, all_hand_wrist[0].shape[-1]))
            raise Exception('Loss for seq-to-seq hasn\'t been implemented yet.')

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.state = None
        self.poses_3d = poses_3d

        
    def num_frames(self):
        count = 0
        for p in self.poses_3d:
            count += p.shape[0]
        return count
    

    def next_pairs(self):
        if self.shuffle:
                pairs = self.random.permutation(self.pairs)
        else:
                pairs = self.pairs
        return 0, pairs


    def next_epoch(self):
        start_idx, pairs = self.next_pairs() # paris: (N_total, 4)
        for b_i in range(start_idx, self.num_batches):
            chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size] # List of len=1 ~ (4,)
            for i, (seq_i, start_frame, end_frame, flip) in enumerate(chunks):
                # Start and end frame index of middle frame
                start = start_frame - self.pad - self.causal_shift
                end = end_frame + self.pad - self.causal_shift
                # Get current chunk from sequence
                seq_img_tensor = self.input_img_tensor[seq_i] # (F, 3, 224, 224)
                seq_3d = self.poses_3d[seq_i]                 # (F, 21, 3)
                seq_vis_flag = self.all_vis_flag[seq_i]       # (F, 21)
                seq_hand_wrist = self.all_hand_wrist[seq_i]   # (F, 3)
                # Compute valid start and end to pad
                start_valid = max(start, 0)
                end_valid = min(end, seq_img_tensor.shape[0])
                pad_left = start_valid - start
                pad_right = end - end_valid
                # If sequence-to-one, only return middle frame's pose3d as target
                if self.seq_to_one:
                        self.batch_input[i] = torch.from_numpy(np.pad(seq_img_tensor[start_valid:end_valid].numpy(), 
                                                                      ((pad_left, pad_right), (0, 0), (0, 0), (0,0)), 'edge'))
                        self.batch_3d[i] = seq_3d[start_frame:end_frame]
                        self.batch_vis_flag[i] = seq_vis_flag[start_frame:end_frame]
                        self.batch_hand_wrist[i] = seq_hand_wrist[start_frame:end_frame]
                else:
                    self.batch_input[i] = torch.from_numpy(np.pad(seq_img_tensor[start_valid:end_valid].numpy(), 
                                                                  ((pad_left, pad_right), (0, 0), (0, 0), (0,0)), 'edge'))
                    self.batch_3d[i] = torch.from_numpy(np.pad(seq_3d[start_valid:end_valid].numpy(), 
                                                               ((pad_left, pad_right), (0, 0), (0, 0)), 'edge'))
                    self.batch_vis_flag[i] = torch.from_numpy(np.pad(seq_vis_flag[start_valid:end_valid].numpy(),
                                                                     ((pad_left, pad_right), (0, 0)), 'edge'))
                    self.batch_hand_wrist[i] = torch.from_numpy(np.pad(seq_hand_wrist[start_valid:end_valid].numpy(),
                                                                       ((pad_left, pad_right), (0, 0)), 'edge'))

            yield self.batch_input[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_vis_flag[:len(chunks)], self.batch_hand_wrist[:len(chunks)]