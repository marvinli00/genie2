import torch
import numpy as np
from abc import ABC, abstractmethod

from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.feat_utils import (
    convert_np_features_to_tensor,
    convert_tensor_features_to_numpy,
    batchify_np_features,
    debatchify_np_features,
    create_empty_np_features,
    save_np_features_to_pdb
)
from genie.sampler.unconditional import UnconditionalSampler
# from genie.sampler.twisted_diffusion.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights
# from genie.sampler.twisted_diffusion.feynman_kac_pf import smc_FK
import numpy as np  
import os
from Bio.PDB import PDBParser
class SMCSampler(UnconditionalSampler):
    """
    SMCSampler for protein structure generation using Sequential Monte Carlo,
    now integrating smc_FK from twisted_diffusion_sampler.
    """
    
    # def multinomial_resample(self,particles, twisting_values, log_weights):
    #     # weights sum to 1
    #     # Draw K indices according to weights
        
        
    #     log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
    #     weights = torch.exp(log_weights)
    #     indices = torch.multinomial(weights, num_samples=particles.shape[0], replacement=True)
        
    #     # Select particles and their twisting values using these indices
    #     #I dont want to change the gradient graph, what can I do ?
    #     new_particles = particles[indices]
    #     new_twisting_values = twisting_values[indices]
        
    #     return new_particles, new_twisting_values
    def multinomial_resample(self, particles, twisting_values, log_weights):
        # Normalize log weights
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        weights = torch.exp(log_weights)
        
        # Draw indices according to weights and detach them
        indices = torch.multinomial(weights, num_samples=particles.shape[0], replacement=True).detach()
        
        # Use index_select on the transformation components
        new_trans = torch.index_select(particles.trans, 0, indices)
        new_rots = torch.index_select(particles.rots, 0, indices)
        new_twisting_values = torch.index_select(twisting_values, 0, indices)
        
        # Create new transformation object with selected components
        new_particles = T(new_rots, new_trans)
        
        return new_particles, new_twisting_values
    def get_denoised_trans(self, ts, timesteps, features, z_pred):
        """
        Get denoised trans directly (predicting x₀ from xₜ)
        
        Args:
            ts: Current transformation (contains trans)
            timesteps: Current diffusion timesteps
            features: Model features
            z_pred: Predicted noise from the model
        """
        # Get alphas_cumprod for current timesteps
        alphas_cumprod = self.model.alphas_cumprod[timesteps]
        
        # Compute x₀ prediction directly
        denoised_trans = (
            (1. / torch.sqrt(alphas_cumprod)).view(-1, 1, 1) * ts.trans - 
            ((1. / torch.sqrt(alphas_cumprod) - 1)).view(-1, 1, 1) * z_pred
        )
        
        # Apply residue mask
        denoised_trans = denoised_trans * features['residue_mask'].unsqueeze(-1)
        
        return denoised_trans
    
    def scale_back(self,x,timesteps):
        return x*(self.model.one_minus_alphas_cumprod[timesteps][0].float())
    
    def prior_weight(self, params):
        # TODO: implement more than one batch size
        # uniform prior
        return torch.ones(params['num_samples']*params["num_particles"], device=self.device)/(params['num_samples']*params["num_particles"])
    
    
    def gaussian_resample(self, mean, std, reference_shape):
        # xk is the current particles
        # mean is the mean of the gaussian, I want to also shift the mean by the current particles
        # std is the standard deviation of the gaussian
        # return the new particles
        return mean + std * torch.randn_like(reference_shape)
    
    def gaussian_prob_tilde(self, x, mean, std, return_log_prob=False):
        if return_log_prob:
            return -0.5 * ((x - mean) / std) ** 2 - torch.log(std * torch.sqrt(2 * torch.tensor(torch.pi)))
        else:
            return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(2 * torch.tensor(torch.pi)))
    
    def reshape_trans_for_motif(self, trans, batch_size = 1, num_particles = 1, reshape_for_diffusion_network = False):
        if not reshape_for_diffusion_network:
            #only change the first dimension
            original_shape = list(trans.shape)
            original_shape[0] = batch_size
            original_shape.insert(1, num_particles)
            return trans.reshape(original_shape)
        else:
            original_shape = list(trans.shape)
            original_shape[0] = batch_size*num_particles
            original_shape.pop(1)
            return trans.reshape(original_shape)
    
    def twisting_function(self, trans, motif_index_mask, motif_target, timesteps):
        #assuming trans.shape = (batch_size, num_particles, sequence_length, 3)
        #motif_index_mask is in the shape of (max_offsets, num_motifs, sequence_length, 3)
        #motif_target is in the shape of (num_motifs, sequence_length, 3)
        score = 0
        for i in range(motif_index_mask.shape[1]):
            motif_index_mask_i = motif_index_mask[:,i,:,:]
            
            #select the first placements for debugging
            motif_index_mask_i = motif_index_mask_i[10:11,:,:]
            
            motif_target_i = motif_target[i]
            
            ts_com_zero = torch.einsum('bkld,old->obkld',trans, motif_index_mask_i)
            ts_com_zero = ts_com_zero.masked_select(motif_index_mask_i[:,None,None,:])
            ts_com_zero = ts_com_zero.reshape(motif_index_mask_i.shape[0],trans.shape[0],trans.shape[1],-1,3)
            ts_com_zero = ts_com_zero - ts_com_zero.mean(dim=-2, keepdim=True) 
            
            #TODO: keep log prob
            #motif_target_i is in the shape of (motif_target_i.shape[0], 3)
            #sum over except for the batch and particle dimensions
            score_i = torch.sum((ts_com_zero - motif_target_i[None,None,None,:]) ** 2, dim=(3,4))


            #TODO: change to log prob function later
            score = score + score_i
        #score = score / motif_index_mask.shape[1]
        score_log_proob_given_motif = torch.logsumexp(score, dim = 0) - torch.log(torch.tensor(score.shape[0], device=self.device))
        #score_log_proob_given_motif = torch.log(score.exp().mean(dim = 0))
        #twisting score log_prob
        score_log_proob_given_motif = -score_log_proob_given_motif/(2*self.model.one_minus_alphas_cumprod[timesteps])
        return score_log_proob_given_motif
    #@torch.compile()
    def center_normalize_motif(self, motif_target):
        motif_target = motif_target - motif_target.mean(dim=0)
        motif_target = motif_target / motif_target.norm(dim=1, keepdim=True)
        return motif_target
    def load_motif_target(self, path = "motif_target.pbd"):
        #motif_target = torch.load(path)
        path = "/home/pc/Documents/protein_design/twisted_diffusion_sampler/protein_exp/motif_scaffolding/targets/5YUI_motif.pdb"
        #write a function to read the motif segments from the pdb file
        def read_motif_segments(pdb_file):
            #read the pdb file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('5YUI', pdb_file)
            return structure

        structure = read_motif_segments(path)
        #leave c-alpha atoms only
        ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA']
        motif_segments = {}
        motif_segments['motif_1'] = ca_atoms[:5]
        motif_segments['motif_2'] = ca_atoms[5:8]
        motif_segments['motif_3'] = ca_atoms[8:]
        motif_coords = {}
        motif_coords['motif_1'] = [atom.get_coord() for atom in motif_segments['motif_1']]
        motif_coords['motif_2'] = [atom.get_coord() for atom in motif_segments['motif_2']]
        motif_coords['motif_3'] = [atom.get_coord() for atom in motif_segments['motif_3']]

        #def a translation matrix to make their centers of mass at the origin
        def center_coordinates(coords):
            """
            Center a list of 3D coordinates by subtracting the mean.
            
            Parameters:
            coords: numpy array of shape (N, 3) or list of [x, y, z] coordinates
            
            Returns:
            numpy array of centered coordinates with same shape as input
            """
            # Convert to numpy array if input is a list
            coords = np.array(coords)#/10
            
            # Calculate mean along each axis
            center = np.mean(coords, axis=0)
            
            # Subtract mean from all points
            centered_coords = coords - center
            
            return centered_coords
        
        motif_target = [center_coordinates(motif_coords['motif_1'])]
        return motif_target
    
    def generate_motif_index_mask(self, motif_target, ts, K_particles):
        trans = ts.trans
        #if we can iterate over the all possible combinations of particles and motif targets, then we can do this
        #otherwise, we need to uniformly sample possible locations for the motif target
        
        #make is in the shape of (offsets, sequence_length, 3)
        
        #code from twisted diffusion
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
        #all_motif_locations is a list of lists of tuples, each tuple is a (start, end) location for a segment
        #np.random.choice works for 1D arrays, we need to change a function to work for 2D arrays
        
        
        # if len(all_motif_locations) > K_particles:
        #     # Convert to numpy array of indices and sample from those
        #     indices = np.random.choice(len(all_motif_locations), K_particles, replace=False)
        #     all_motif_locations = [all_motif_locations[i] for i in indices]
        # else:
        #     # Sample indices with replacement when we need more particles than available locations
        #     indices = np.random.choice(len(all_motif_locations), K_particles, replace=True)
        #     all_motif_locations = [all_motif_locations[i] for i in indices]
        
        #generate a mask for the motif target
        offsets = len(all_motif_locations)
        
        #make is in the shape of (offsets, number_of_motifs, sequence_length, 3)
        shape_of_trans = [offsets, len(all_motif_locations[0]),trans.shape[1], 3]
        motif_index_mask = torch.zeros(shape_of_trans, dtype=torch.bool)
        #motif_index_mask = motif_index_mask.permute(0,3,1,2)
        #Todo check this
        for i in range(offsets):
            for j in range(len(all_motif_locations[i])):
                motif_index_mask[i, j,all_motif_locations[i][j][0]:(all_motif_locations[i][j][1]+1)] = True
        return motif_index_mask.to(self.device)
    
    def log_prob_gaussian(self, x, mean, std):
        return -0.5 * ((x - mean) / std) ** 2 - torch.log(std * torch.sqrt(2 * torch.tensor(torch.pi)))
    
    def q_x_t_minus_1_x_t(self, x_t_minus_1, x_t,z,timesteps):
        score = -1 / (self.model.sqrt_one_minus_alphas_cumprod[timesteps])[0].item() * z
        stds = torch.std(x_t_minus_1, dim = (1,2)).mean(dim = 0)
        mean = x_t_minus_1 - score*stds**2
        return self.log_prob_gaussian(x_t, mean, stds)
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
        number_of_particles = 2
        params["num_particles"] = number_of_particles
        
        rescale_motifs = 0.6
        
        # setup for twisting
        # Create features
        features = convert_np_features_to_tensor(
            batchify_np_features([
                self.create_np_features(params)
                for _ in range(params['num_samples']*params["num_particles"])
            ]),
            self.device
        )
        #repeat along the first (batch) dimension
        #features['atom_positions'] = features['atom_positions'].repeat(params["num_particles"])
        
        
        #create the motif target
        motif_target = self.load_motif_target()
        motif_target = [torch.from_numpy(i).to(self.device) for i in motif_target]
        
        
        # Create positions
        trans = torch.randn_like(features['atom_positions'])
        
        #smc_weights = self.prior_weight(params).to(trans.dtype)
        
        
        
        #pk = smc_weights
        
        
        
        rots = compute_frenet_frames(
            trans,
            features['chain_index'],
            features['residue_mask']
        )
        ts = T(rots, trans)
        
        
        
        #requires grad for the coordinates
        ts.trans.requires_grad_(True)
        
        
        #create the motif index mask
        motif_index_mask = self.generate_motif_index_mask(motif_target, ts, params["num_particles"])
        #wk and pk are in the log prob
        wk_plus_1 = pk_plus_1 = self.twisting_function(self.reshape_trans_for_motif(ts.trans, params['num_samples'], params['num_particles'], reshape_for_diffusion_network = False), 
                                         motif_index_mask, motif_target, self.model.config.diffusion['n_timestep'])
        pk_plus_1 = self.reshape_trans_for_motif(pk_plus_1, params['num_samples'], params['num_particles'], reshape_for_diffusion_network = True)
        wk_plus_1 = self.reshape_trans_for_motif(wk_plus_1, params['num_samples'], params['num_particles'], reshape_for_diffusion_network = True)
        
        # Define steps
        steps = reversed(np.arange(1, self.model.config.diffusion['n_timestep'] + 1))

        # Iterate
        for step in steps:
            trans = ts.trans
            #ts.trans.requires_grad_(True)
            timesteps = torch.Tensor([step] * params['num_samples']*params["num_particles"]).int().to(self.device)
            # resample
            #ts, pk_plus_1 = self.multinomial_resample(ts, pk_plus_1, wk_plus_1)
            
            # Define current diffusion timestep
            

            # Compute noise
            with torch.no_grad():
                temp = self.model.model(ts, timesteps, features)
                z_pred = temp['z']
            
            w_z = (1. - self.model.alphas[timesteps]) / self.model.sqrt_one_minus_alphas_cumprod[timesteps]
            trans_mean = (1. / self.model.sqrt_alphas[timesteps]).view(-1, 1, 1) * (ts.trans - w_z.view(-1, 1, 1) * z_pred)
            trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)
            trans_z = torch.randn_like(ts.trans)
            trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
            trans_mean = trans_mean + params['scale'] * trans_sigma * trans_z
            trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)
            
            
            # if timesteps[0] > 800:
            #     p_tilde_k_t = self.twisting_function(self.reshape_trans_for_motif(trans_mean, params['num_samples'], params['num_particles'], reshape_for_diffusion_network = False), 
            #                                      motif_index_mask, motif_target, timesteps-1)
            #     pk_plus_1 = p_tilde_k_t
            #     trans_z = torch.randn_like(ts.trans)
            #     trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
            #     trans_mean = trans_mean + params['scale'] * trans_sigma * trans_z
            #     trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)
            #     rots = compute_frenet_frames(
            #         trans_mean,
            #         features['chain_index'],
            #         features['residue_mask']
            #     )
            #     ts = T(rots.detach(), trans_mean)
                
            #     continue
            
            #Algorithm 1 conditional score approximation
            #set trans also to the same scale as the denoised_trans
            #denoised_trans = self.get_denoised_trans(ts, timesteps, features, z_pred)
            #sk = (denoised_trans - ts.trans)/self.model.one_minus_alphas_cumprod[timesteps].reshape(-1,1,1)
            
            
            
            #short_cut I guess?
            #TODO: Gaussian is symmetric, so we can take off the negative sign
            
            #variance prevervation ddpm
            sk = (self.model.sqrt_alphas_cumprod[timesteps][0]*trans_mean - ts.trans)/self.model.one_minus_alphas_cumprod[timesteps][0]
            #sk = -z_pred
            #All gradients are computed with respect to the twisting values
            pk_reshaped = pk_plus_1.reshape(-1, 1)
            grad_outputs = torch.ones_like(pk_reshaped)

            #TODO: should I set back the ts.trans to the original scale?
            #TODO: what about last scale. should I? Maybe not
            # if step == self.model.config.diffusion['n_timestep']:
            #     x_plus_1 = self.scale_back(ts.trans, timesteps) 
            # else:
            #     x_plus_1 = self.scale_back(ts.trans, timesteps+1)
            
            grad_x_t_plus_1 = torch.autograd.grad(pk_reshaped, ts.trans, grad_outputs = grad_outputs, create_graph=False, allow_unused=True)[0]
            #grad_x_t_plus_1 = 0
            
            pk_reshaped = pk_reshaped.detach()
            ts.trans = ts.trans.detach()
            sk = sk + grad_x_t_plus_1
            
            #resample
            
            #mean_shift = sk*self.model.one_minus_alphas_cumprod[timesteps]
            gaussian_std = self.model.sqrt_betas[timesteps][0]
            gaussian_mean_shift = trans/self.model.sqrt_alphas[timesteps][0] + sk*gaussian_std ** 2/self.model.sqrt_alphas[timesteps][0]
            
            #TODO: proposal
            trans_next_step = self.gaussian_resample(gaussian_mean_shift, gaussian_std, reference_shape=ts.trans)
            
            #TODO: When should I stop grad?
            
            # twisting_function(self, trans, motif_index_mask, motif_target, timesteps):
            
            #compute the twisting values for the next step
            #TODO: done
            p_tilde_k_t = self.twisting_function(self.reshape_trans_for_motif(trans_next_step, params['num_samples'], params['num_particles'], reshape_for_diffusion_network = False), 
                                                 motif_index_mask, motif_target, timesteps-1)
            pk_plus_1 = p_tilde_k_t
            # #update the resampling weights
            # wk = self.twisting_function(ts, motif_index_mask, motif_target, timesteps)
            
            p_tilde_k_t = self.reshape_trans_for_motif(p_tilde_k_t, params['num_samples'], params['num_particles'], reshape_for_diffusion_network = True)
            
            
            #line 9 of the algorithm
            p_theta = self.q_x_t_minus_1_x_t(trans_next_step, ts.trans,z_pred,timesteps)
            p_theta = p_theta.mean(dim = (1,2))            
            
            p_theta_y = self.gaussian_prob_tilde(ts.trans, gaussian_mean_shift, gaussian_std)
            p_theta_y = p_theta_y.mean(dim = (1,2))
            
            mean_plus_1 = grad_x_t_plus_1*gaussian_std**2+ts.trans
            std_plus_1 = gaussian_std
            
            p_theta_y_plus_1 = self.log_prob_gaussian(trans_next_step, mean_plus_1, std_plus_1)
            p_theta_y_plus_1 = p_theta_y_plus_1.mean(dim = (1,2))
            wk_plus_1 = p_theta*p_tilde_k_t/(p_theta_y*p_theta_y_plus_1)

            # Sample
            if step == 1:
                # Compute rotations
                rots_mean = compute_frenet_frames(
                    trans_next_step,
                    features['chain_index'],
                    features['residue_mask']
                )

                # Compute frames
                ts = T(rots_mean.detach(), trans_mean.detach())

            else:

                # Compute rotations
                rots = compute_frenet_frames(
                    trans_next_step,
                    features['chain_index'],
                    features['residue_mask']
                )
                ts = T(rots.detach(), trans_next_step)

                # Compute translations
                # trans_z = torch.randn_like(ts.trans)
                # trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
                # trans = trans_mean + params['scale'] * trans_sigma * trans_z
                # trans = trans * features['residue_mask'].unsqueeze(-1)

                # # Compute rotations
                # rots = compute_frenet_frames(
                #     trans,
                #     features['chain_index'],
                #     features['residue_mask']
                # )

                # Compute frames
                #ts = T(rots.detach(), trans.detach())

        # Postprocess
        features['atom_positions'] = ts.trans.detach().cpu()
        np_features = convert_tensor_features_to_numpy(features)
        list_np_features = debatchify_np_features(np_features)

        return list_np_features

    ###############################
    ###   Required Parameters   ###
    ###############################

    def add_required_parameter(self, name):
        """
        Add an additional required parameter.

        Args:
            name:
                Name of the required parameter to be added.
        """
        self.required.append(name)

    def validate_parameters(self, params):
        """
        Validate if all required parameters are present in the dictionary
        of sampling parameters.

        Args:
            params:
                A dictionary of sampling parameters. Required parameters include
                    -    scale: sampling noise scale
                    -    outdir: output directory
                    -    num_samples: number of samples to generate (in a batch)
                    -    prefix: prefix for filenames of generated structures
                    -    offset: offset for distinguishing between batches
                    -    user-defined parameters (by calling add_required_parameter).
        """
        for name in self.required:
            if name not in params:
                return False
        return True
