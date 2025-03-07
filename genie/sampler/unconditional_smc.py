import os

from genie.sampler.base import BaseSampler
from genie.utils.feat_utils import (
    create_empty_np_features,
    save_np_features_to_pdb
)
import torch
import numpy as np
from abc import ABC, abstractmethod
from genie.sampler.unconditional import UnconditionalSampler
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.feat_utils import (
    convert_np_features_to_tensor,
    convert_tensor_features_to_numpy,
    batchify_np_features,
    debatchify_np_features
)
import math
from torch.distributions.normal import Normal
from Bio.PDB import PDBParser
from functools import partial
import wandb
from genie.sampler.utils import process_pdb, load_motif_target
def compute_ess(w, dim=0):
    ess = (w.sum(dim=dim))**2 / torch.sum(w**2, dim=dim)
    return ess

def compute_ess_from_log_w(log_w, dim=0):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)

def normalize_weights(log_weights, dim=0):
    return torch.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights
def normalize_log_weights_everything(log_weights_list, dim):
    return [normalize_log_weights(log_weights, dim) for log_weights in log_weights_list]

class SMCSampler(UnconditionalSampler):

    def setup(self):
        """
        Set up by adding additional required parameters.
        """
        self.add_required_parameter('length')

    def on_sample_start(self, params):
        """
        Set up an output directory if necessary before sampling starts. The directory 
        is named 'pdbs', where each file stores the generated structure in a PDB format.

        Args:
            params:
                A dictionary of sampling parameters. Required parameters include
                    -    scale: sampling noise scale
                    -    outdir: output directory
                    -    num_samples: number of samples to generate (in a batch)
                    -    prefix: prefix for filenames of generated structures
                    -    offset: offset for distinguishing between batches
                    -    length: target sequence length.
        """
        pdbs_dir = os.path.join(params['outdir'], 'pdbs')
        if not os.path.exists(pdbs_dir):
            os.makedirs(pdbs_dir)

    def create_np_features(self, params):
        """
        Creates a feature dictionary in numpy (without padding or batching operations).

        Args:
            params:
                A dictionary of sampling parameters. Required parameters include
                    -    scale: sampling noise scale
                    -    outdir: output directory
                    -    num_samples: number of samples to generate (in a batch)
                    -    prefix: prefix for filenames of generated structures
                    -    offset: offset for distinguishing between batches
                    -    length: target sequence length.

        Returns:
            A feature dictionary containing information on an input structure 
            of length N, including
                -    aatype: 
                        [N, 20] one-hot encoding on amino acid types. All amino acid
                        types are set to 'ALA' since Genie 2 is sequence-agnostic.
                -    num_chains: 
                        [1] number of chains in the structure
                -    num_residues: 
                        [1] number of residues in the structure
                -    num_residues_per_chain: 
                        [1] an array of number of residues by chain
                -    atom_positions: 
                        [N, 3] an array of Ca atom positions. Atom positions of 
                        all residues are default to the origin. 
                -    residue_mask: 
                        [N] residue mask to indicate which residue position is masked
                -    residue_index: 
                        [N] residue index (started from 0)
                -    chain_index: 
                        [N] chain index (started from 0)
                -    fixed_sequence_mask: 
                        [N] mask to indicate which residue contains conditional
                        sequence information
                -    fixed_structure_mask: 
                        [N, N] mask to indicate which pair of residues contains
                        conditional structural information
                -    fixed_group:
                        [N] group index to indicate which group the residue belongs to
                        (useful for specifying multiple functional motifs)
                -    interface_mask:
                        [N] deprecated and set to all zeros.
        """
        return create_empty_np_features([params['length']])

    def on_sample_end(self, params, list_np_features):
        """
        Save generated structures (in a directory named 'pdbs').

        Args:
            params:
                A dictionary of sampling parameters. Required parameters include
                    -    scale: sampling noise scale
                    -    outdir: output directory
                    -    num_samples: number of samples to generate (in a batch)
                    -    prefix: prefix for filenames of generated structures
                    -    offset: offset for distinguishing between batches
                    -    length: target sequence length.
            list_np_features:
                A list of feature dictionaries, each of which has padding removed and 
                stores the following information on a generated structure of length N
                    -    aatype: 
                            [N, 20] one-hot encoding on amino acid types. All amino acid
                            types are set to 'ALA' since Genie 2 is sequence-agnostic.
                    -    num_chains: 
                            [1] number of chains in the structure
                    -    num_residues: 
                            [1] number of residues in the structure
                    -    num_residues_per_chain: 
                            [1] an array of number of residues by chain
                    -    atom_positions: 
                            [N, 3] an array of Ca atom positions
                    -    residue_mask: 
                            [N] residue mask to indicate which residue position is masked
                    -    residue_index: 
                            [N] residue index (started from 0)
                    -    chain_index: 
                            [N] chain index (started from 0)
                    -    fixed_sequence_mask: 
                            [N] mask to indicate which residue contains conditional
                            sequence information
                    -    fixed_structure_mask: 
                            [N, N] mask to indicate which pair of residues contains
                            conditional structural information
                    -    fixed_group:
                            [N] group index to indicate which group the residue belongs to
                            (useful for specifying multiple functional motifs)
                    -    interface_mask:
                            [N] deprecated and set to all zeros.
        """
        for i, np_features in enumerate(list_np_features):
            name = '{}_{}'.format(params['prefix'], params['offset'] + i)
            output_pdb_filepath = os.path.join(
                params['outdir'], 'pdbs', 
                '{}.pdb'.format(name)
            )
            save_np_features_to_pdb(np_features, output_pdb_filepath)

    def generate_motif_index_mask(self, motif_target, trans):
        def get_all_motif_locations(L, segment_lengths, max_offsets=1000, first_call=True):
            """get_all_motif_locations returns of all possible starting and ending locations segments of length segment_lengths
            such that not of the segments overlap, the smallest starting location at least 0, and the largest ending location is at most L-1.

            The function works recursively.  First, it computes all possible starting and ending locations for the first segment.
            Then, it computes all possible starting and ending locations for the second segment, given the starting and ending locations

            Args:
                L: int, length of sequence
                segment_lengths: list of ints, length of each segment
                max_offsets: int, maximum number of motif offsets to return

            Returns:
                all_motif_locations: list of lists of tuples, each tuple is a (start, end) location for a segment
            """
            st_0_min = 0
            st_0_max = L - sum(segment_lengths)
            all_motif_locations = []
            for st_0 in range(st_0_min, st_0_max+1):
                end_0 = st_0 + segment_lengths[0] - 1

                # base case
                if len(segment_lengths) == 1:
                    all_motif_locations.append(([st_0, end_0],))
                else:
                    remaining_length = L - (end_0 + 1)
                    all_later_motif_locs = get_all_motif_locations(
                        remaining_length, segment_lengths[1:], max_offsets, first_call=False)
                    for later_motif_locs in all_later_motif_locs:
                        later_motif_locs  = [(st + end_0 + 1, end + end_0 + 1) for st, end in later_motif_locs]
                        all_motif_locations.append(
                            [(st_0, end_0)] + later_motif_locs
                        )

            if len(all_motif_locations) > max_offsets and first_call:
                # downsampling from len(all_motif_locations) to max_offsets offsets
                N = len(all_motif_locations)
                idcs = np.random.choice(N, max_offsets, replace=False)
                all_motif_locations = [all_motif_locations[idx] for idx in idcs]

            return all_motif_locations
        length_of_motifs = [len(i) for i in motif_target]
        
        
        max_offsets = 1000
        all_motif_locations = get_all_motif_locations(trans.shape[1], length_of_motifs, max_offsets = max_offsets)
        offsets = len(all_motif_locations)
        
        #make is in the shape of (offsets, number_of_motifs, sequence_length, 3)
        shape_of_trans = [offsets, len(all_motif_locations[0]),trans.shape[1], 3]
        motif_index_mask = torch.zeros(shape_of_trans, dtype=torch.bool)
        #motif_index_mask = motif_index_mask.permute(0,3,1,2)
        for i in range(offsets):
            for j in range(len(all_motif_locations[i])):
                if self.selected_motif_location_index is not None:
                    if i == self.selected_motif_location_index:
                        self.run.log({f"motif_index_mask_{i}_{j}": all_motif_locations[i][j]})
                motif_index_mask[i, j,all_motif_locations[i][j][0]:(all_motif_locations[i][j][1]+1)] = True
        return motif_index_mask.to(self.device)
    
    
    def log_normal_density(self, sample, mean, var):
        return Normal(loc=mean, scale=torch.sqrt(var)).log_prob(sample)
    
    def systematic_resampling(self,particles, weights):
        """
        Perform systematic resampling on a set of particles based on their weights.
        
        Args:
            particles (numpy.ndarray): Array of particles to be resampled (N x D)
            weights (numpy.ndarray): Array of particle weights (N,)
            
        Returns:
            tuple:
                - numpy.ndarray: Resampled particles
                - numpy.ndarray: New uniform weights
                - numpy.ndarray: Indices of selected particles
        """
        N = len(weights)
        # Normalize weights
        
        def normalize_weights(weights):
            return weights / torch.sum(weights)
        
        weights = normalize_weights(weights)
        
        # Calculate cumulative sum of weights
        cumsum = torch.cumsum(weights, dim = 0)
        
        # Generate systematic noise (one random number)
        u = torch.distributions.Uniform(low=0.0, high=1.0/N).sample()
        #u = np.random.uniform(0, 1/N)
        
        # Generate points for systematic sampling
        points = torch.zeros(N)
        for i in range(N):
            points[i] = u + i/N
        
        # Initialize arrays for results
        indexes = torch.zeros(N, dtype=int)
        cumsum = torch.cat([torch.tensor([0.0], device = self.device), cumsum])  # Add 0 at the beginning for easier indexing
        
        # Perform systematic resampling
        i, j = 0, 0
        while i < N:
            while points[i] > cumsum[j+1]:
                j += 1
            indexes[i] = j
            i += 1
        
        # Resample particles and reset weights
        resampled_particles = particles[indexes]
        #new_weights = torch.ones(N, device = self.device) / N
        #log new_weights
        new_weights = torch.zeros(N, device = self.device)
        return resampled_particles, new_weights, indexes   
    
    
    def get_xstart_var(self,alphas_cumprod_t, tausq_=0.05,var_type = 6):
        
        sigmasq_ = (1-alphas_cumprod_t) / alphas_cumprod_t
        if var_type == 1:
            return sigmasq_ 
        elif var_type == 2: # pseudoinverse-guided paper https://openreview.net/forum?id=9_gsMA8MRKQ 
            tausq_ = 1.0 
            return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
            #return (1 - alphas_cumprod_t) 
        elif var_type == 5: 
            tausq_ = 0.30 
            return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
        elif var_type == 3: # DPS paper https://arxiv.org/abs/2209.14687 
            return None  
        elif var_type == 4: # pseudoinverse-guided paper -- the actual implementation, see their Alg.1 
            return beta_t  / np.sqrt(alphas_cumprod_t) 
        elif var_type == 6: # freely specify tausq_
            tausq_ = tausq_ 
            return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
    def twisting_function(self, trans, motif_index_mask, motif_target, timesteps):
        # Initialize the total score to 0
        score = 0
        normalized_variance = self.get_xstart_var(self.model.alphas_cumprod[timesteps], var_type = 6, tausq_ = 0.012)
        # Loop through each motif (supporting multiple motifs)
        for i in range(motif_index_mask.shape[1]):
            motif_index_mask_i = motif_index_mask[:,i,:,:]
            
            # For debugging, select only a specific subset of placements
            # Here, it selects only the 15th placement position for focused evaluation
            #motif_index_mask_i = motif_index_mask_i[15:16,:,:]
            
            # Get the target coordinates for this specific motif
            motif_target_i = motif_target[i]
            
            ts_com_zero = torch.einsum('old,bkld->obkld', motif_index_mask_i, trans)
            
            ts_com_zero = ts_com_zero.masked_select(motif_index_mask_i[:,None,None,:])
            ts_com_zero = ts_com_zero.reshape(motif_index_mask_i.shape[0], trans.shape[0], trans.shape[1], -1, 3)
            ts_com_zero = ts_com_zero - ts_com_zero.mean(dim=-2, keepdim=True)
            
            #Gaussian log probability: -||x-μ||²/(2σ²)
            score_i = -torch.sum((ts_com_zero - motif_target_i[None,None,None,:]) ** 2, dim=(3,4)) / (2*normalized_variance)

            # Log the mean squared distance for monitoring/debugging
            self.run.log({f"distances_of_motif_{i}": ((ts_com_zero - motif_target_i[None,None,None,:])**2).mean()})

            # Add this motif's score to the total score
            # Multiple motifs' scores are summed, giving equal weight to each motif
            score = score + score_i
        score_log_proob_given_motif = torch.logsumexp(score, dim=0) - torch.log(torch.tensor(score.shape[0], device=self.device))
        
        # score_log_prob_given_motif is (1, P)
        
        # Return the final log probability score
        return score_log_proob_given_motif.unsqueeze(0)
    
    def compute_ess_softmax(self,log_weights):
        #softmax
        weights = torch.nn.functional.softmax(log_weights, dim = 0)
        return 1/torch.sum(weights**2)
    
    def _sample(self, params):
        """
        Sampling structures given input sampling parameters.

        Args:
            params:
                A dictionary of sampling parameters. Required parameters include
                    -    scale: sampling noise scale
                    -    outdir: output directory
                    -    num_samples: number of samples to generate (in a batch)
                    -    prefix: prefix for filenames of generated structures
                    -    offset: offset for distinguishing between batches
                    -    user-defined parameters (by calling add_required_parameter).

        Returns:
            list_np_features:
                A list of feature dictionaries, each of which has padding removed and 
                stores the following information on a generated structure of length N
                    -    aatype: 
                            [N, 20] one-hot encoding on amino acid types. All amino acid
                            types are set to 'ALA' since Genie 2 is sequence-agnostic.
                    -    num_chains: 
                            [1] number of chains in the structure
                    -    num_residues: 
                            [1] number of residues in the structure
                    -    num_residues_per_chain: 
                            [1] an array of number of residues by chain
                    -    atom_positions: 
                            [N, 3] an array of Ca atom positions
                    -    residue_mask: 
                            [N] residue mask to indicate which residue position is masked
                    -    residue_index: 
                            [N] residue index (started from 0)
                    -    chain_index: 
                            [N] chain index (started from 0)
                    -    fixed_sequence_mask: 
                            [N] mask to indicate which residue contains conditional
                            sequence information
                    -    fixed_structure_mask: 
                            [N, N] mask to indicate which pair of residues contains
                            conditional structural information
                    -    fixed_group:
                            [N] group index to indicate which group the residue belongs to
                            (useful for specifying multiple functional motifs)
                    -    interface_mask:
                            [N] deprecated and set to all zeros.
        """
        self.run = wandb.init(project="protein_design", name="test001")
        
        params['num_samples'] = 4
        self.run.log({"num_samples": params['num_samples']})
        motif_target = load_motif_target(index = 26)
        #motif_target = [motif_target[0]]
        motif_target = [torch.from_numpy(i).to(self.device) for i in motif_target]
        self.selected_motif_location_index = None
    
        # Create features
        features = convert_np_features_to_tensor(
            batchify_np_features([
                self.create_np_features(params)
                for _ in range(params['num_samples'])
            ]),
            self.device
        )

        # Create frames
        trans = torch.randn_like(features['atom_positions'])
        
        #log 
        log_proposal = self.log_normal_density(trans, torch.tensor(0, device = self.device), torch.tensor(1, device = self.device)).sum(dim = (1,2))
        
        log_proposal_tracker = []
        log_proposal_tracker.append(log_proposal)
        
        #for ess, tracing weights
        ess_tracker = []
        #ess_tracker.append(
        
        #initialize weights
        log_w_prev_accumulated = torch.log(torch.ones_like(log_proposal, device = self.device))
        
        
        motif_index_mask = self.generate_motif_index_mask(motif_target, trans)
        rots = compute_frenet_frames(
            trans,
            features['chain_index'],
            features['residue_mask']
        )
        ts = T(rots, trans)

        # Define steps
        steps = reversed(np.arange(1, self.model.config.diffusion['n_timestep'] + 1))
        
        twisting_function = partial(self.twisting_function, motif_index_mask = motif_index_mask, motif_target = motif_target)
        
        torch_default_dtype = torch.get_default_dtype()
        #write a function change model weights to double
        def change_model_weights_to_double(model):
            for param in model.parameters():
                param.data = param.data.to(torch_default_dtype)
        change_model_weights_to_double(self.model)
        
        #make value in features double when they are float before
        def change_features_to_double(features):
            for key in features:
                if isinstance(features[key], torch.Tensor):
                    # if it is float, change it to double
                    if features[key].dtype == torch.float32:
                        features[key] = features[key].to(torch_default_dtype)
        change_features_to_double(features)
        
        # Iterate
        for step in steps:
            ts.trans.requires_grad = True
            # Define current diffusion timestep
            timesteps = torch.Tensor([step] * params['num_samples']).int().to(self.device)

            # Compute noise
            # with torch.no_grad():
            z_pred = self.model.model(ts, timesteps, features)['z']
   
            #find E[x_0|x_t, t] unconditional
            untwisted_predicted_x_start = (ts.trans - self.model.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1) * z_pred) / self.model.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
            
            #compute log p(y|x_t, t) := log N(y; x_0, sigma_t^2 I)
            log_prob = twisting_function(untwisted_predicted_x_start.unsqueeze(0),timesteps = step).squeeze()
            
   
            grad_pk_with_respect_to_x_t = torch.autograd.grad(log_prob.mean(), ts.trans)[0]
            #rescale mean back to the original scale
            grad_pk_with_respect_to_x_t = grad_pk_with_respect_to_x_t*params['num_samples']
            with torch.no_grad():
                alpha = 0.012
                # |grad_pk_with_respect_to_x_t|_F
                norm_grad = grad_pk_with_respect_to_x_t.norm()
                #regularize gradient to prevent gradient explosion
                grad_pk_with_respect_to_x_t = grad_pk_with_respect_to_x_t*alpha*norm_grad/(alpha+norm_grad)
            if step >= 50:               
                twisted_predicted_x_start = untwisted_predicted_x_start + grad_pk_with_respect_to_x_t
            else:
                ##uncondition sample at the very last steps for stability
                twisted_predicted_x_start = untwisted_predicted_x_start
            
            posterior_mean_coef1 = self.model.sqrt_alphas_cumprod[timesteps-1]*self.model.betas[timesteps]/(self.model.one_minus_alphas_cumprod[timesteps])
            posterior_mean_coef2 = self.model.sqrt_alphas[timesteps]*(1.0-self.model.alphas_cumprod[timesteps-1])/(self.model.one_minus_alphas_cumprod[timesteps])
            #reshape
            posterior_mean_coef1 = posterior_mean_coef1.view(-1, 1, 1)
            posterior_mean_coef2 = posterior_mean_coef2.view(-1, 1, 1)
            #find u(x_t-1) = E[x_t-1|x_t, (x_0, y),t]
            trans_mean = (posterior_mean_coef1 * twisted_predicted_x_start + 
                              posterior_mean_coef2 * ts.trans)
   
            with torch.no_grad():
                trans_mean_untwisted = (posterior_mean_coef1 * untwisted_predicted_x_start + posterior_mean_coef2 * ts.trans)
            # w_z = (1. - self.model.alphas[timesteps]) / self.model.sqrt_one_minus_alphas_cumprod[timesteps]
            # trans_mean = (1. / self.model.sqrt_alphas[timesteps]).view(-1, 1, 1) * (ts.trans - w_z.view(-1, 1, 1) * z_pred)
            # trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)
            
   
            # Sample
            if step == 1:

                # Compute rotations
                rots_mean = compute_frenet_frames(
                    trans_mean,
                    features['chain_index'],
                    features['residue_mask']
                )

                # Compute frames
                ts = T(rots_mean.detach(), trans_mean.detach())

            else:
                with torch.no_grad():
                    # Compute translations
                    trans_z = torch.randn_like(ts.trans)
                    trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
                    trans = trans_mean + params['scale'] * trans_sigma * trans_z
                    trans = trans * features['residue_mask'].unsqueeze(-1)

    
                    #get p~^(t+1)_k
                    log_proposal = log_proposal_tracker.pop().squeeze()
                    log_proposal_tracker.append(log_prob)
                    #get p~^(t)_k
                    log_potential_xt = log_prob
                

                    # Find p(xt_k|xt+1_k) - the reverse transition probability
                    log_reverse_transition = self.log_normal_density(trans, trans_mean_untwisted, trans_sigma.pow(2)).sum(dim = (1,2))

                    # Find p~(xt_k|xt+1_k,y) - the twisted reverse transition
                    log_twisted_transition = self.log_normal_density(trans, trans_mean, trans_sigma.pow(2)).sum(dim = (1,2))
                    temp = log_reverse_transition - log_twisted_transition
                    # Calculate importance weight
                    log_target = log_reverse_transition + log_potential_xt - log_twisted_transition  
                    #unnormalize log_w
                    log_w = log_target - log_proposal
                    log_w_accumulated = log_w + log_w_prev_accumulated
                    
                    ess =  compute_ess_from_log_w(log_w_accumulated)
                    
                    # ess = self.compute_ess(log_w_accumulated)
                    ess_tracker.append(ess.detach().cpu().numpy())
                    self.run.log({"ess": ess})
                    #resample when ess is too low (50% of num_samples)
                    if ess < 0.5*params['num_samples']:
                        weights = torch.nn.functional.softmax(log_w_accumulated, dim = 0)
                        #resample
                        trans, log_w_prev_accumulated, indexes = self.systematic_resampling(trans, weights)
                        log_proposal_tracker[0] = log_proposal_tracker[0][indexes]
                    else:
                        #log_w = normalize_log_weights(log_w, dim=0)
                        log_w_prev_accumulated = normalize_log_weights(log_w_accumulated, dim=0) + torch.log(torch.tensor(params['num_samples'], device=self.device))
                    # Compute rotations
                    rots = compute_frenet_frames(
                        trans,
                        features['chain_index'],
                        features['residue_mask']
                    )

                    # Compute frames
                    ts = T(rots.detach(), trans.detach())

        # Postprocess
        features['atom_positions'] = ts.trans.detach().cpu()
        np_features = convert_tensor_features_to_numpy(features)
        list_np_features = debatchify_np_features(np_features)
        self.run.finish()
        return list_np_features