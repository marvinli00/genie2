import os

from genie.sampler.base import BaseSampler
from genie.utils.feat_utils import (
    create_empty_np_features,
    save_np_features_to_pdb
)
import torch
import numpy as np

from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.feat_utils import (
    convert_np_features_to_tensor,
    convert_tensor_features_to_numpy,
    batchify_np_features,
    debatchify_np_features
)
from genie.sampler.unconditional import UnconditionalSampler
# from genie.sampler.twisted_diffusion.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights
# from genie.sampler.twisted_diffusion.feynman_k        ac_pf import smc_FK
import numpy as np  
import os
from Bio.PDB import PDBParser
from functools import partial
import wandb
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
    def load_motif_target(self, path = "motif_target.pbd"):
        #motif_target = torch.load(path)
        path = "/hai/scratch/mli89/protein_design/twisted_diffusion_sampler/protein_exp/motif_scaffolding/targets/6E6R_long_motif.pdb"
        #path = "/home/pc/Documents/protein_design/twisted_diffusion_sampler/protein_exp/motif_scaffolding/targets/5YUI_motif.pdb"
        
        
        #write a function to read the motif segments from the pdb file
        def read_motif_segments(pdb_file):
            #read the pdb file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('5YUI', pdb_file)
            return structure

        structure = read_motif_segments(path)
        
        
        if "5YUI" in path:
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
        elif "6E6R" in path:
            #leave c-alpha atoms only
            ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA']
            motif_coords = {}
            motif_coords['motif_1'] = [atom.get_coord() for atom in ca_atoms]
        else:
            raise ValueError(f"Unknown motif target: {path}")

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
        #load more motifs
        motif_target = []
        for motif in motif_coords:
            motif_target.append(center_coordinates(motif_coords[motif]))
        return motif_target
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
    def normalize_weights(self, weights):
        """
        Normalize weights to sum to 1.
        
        Args:
            weights (numpy.ndarray): Array of unnormalized weights
            
        Returns:
            numpy.ndarray: Normalized weights that sum to 1
        """
        return weights / np.sum(weights)

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
        weights = self.normalize_weights(weights)
        
        # Calculate cumulative sum of weights
        cumsum = np.cumsum(weights)
        
        # Generate systematic noise (one random number)
        u = np.random.uniform(0, 1/N)
        
        # Generate points for systematic sampling
        points = np.zeros(N)
        for i in range(N):
            points[i] = u + i/N
        
        # Initialize arrays for results
        indexes = np.zeros(N, dtype=int)
        cumsum = np.append(0, cumsum)  # Add 0 at the beginning for easier indexing
        
        # Perform systematic resampling
        i, j = 0, 0
        while i < N:
            while points[i] > cumsum[j+1]:
                j += 1
            indexes[i] = j
            i += 1
        
        # Resample particles and reset weights
        resampled_particles = particles[indexes]
        new_weights = np.ones(N) / N
        
        return resampled_particles, new_weights, indexes    
    
    
    
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
            
            #ts_com_zero = torch.einsum('bkld,old->obkld',trans, motif_index_mask_i)
            ts_com_zero = torch.einsum('old,bkld->obkld', motif_index_mask_i,trans)
            ts_com_zero = ts_com_zero.masked_select(motif_index_mask_i[:,None,None,:])
            ts_com_zero = ts_com_zero.reshape(motif_index_mask_i.shape[0],trans.shape[0],trans.shape[1],-1,3)
            ts_com_zero = ts_com_zero - ts_com_zero.mean(dim=-2, keepdim=True) 
            
            #TODO: keep log prob
            #motif_target_i is in the shape of (motif_target_i.shape[0], 3)
            #sum over except for the batch and particle dimensions
            
            
            #calculate the log prob of p_tidle_given_x_t for each fixed location
            score_i = -torch.sum((ts_com_zero - motif_target_i[None,None,None,:]) ** 2, dim=(3,4))/(2*self.model.one_minus_alphas_cumprod[timesteps])

            self.run.log({f"distances_of_motif_{i}": ((ts_com_zero- motif_target_i[None,None,None,:])**2).mean()})


            #TODO: change to log prob function later
            score = score + score_i
        #score = score / motif_index_mask.shape[1]
        
        with torch.no_grad():
            # Store all distributions over time
            softmax_scores_over_time = []

            # During training/inference:
            softmax_score = torch.softmax(score, dim=0)[:,0,0]
            softmax_scores_over_time.append(softmax_score.cpu().numpy())
            self.softmax_tracking.append(softmax_score.cpu().numpy())
            # Log as multiple lines to see the evolution
            # self.run.log({
            #     "softmax_distribution_over_time": wandb.plot.line_series(
            #         xs=np.arange(score.shape[0]),  # indices
            #         ys=softmax_scores_over_time,   # multiple distributions
            #         title="Softmax Distribution Evolution",
            #         xname="Index",
            #     )
            # })
        
        #sum over the possible motif placements, change log prob to prob and sum over. take log of the sum at the end. 1/|M| becomes -log|M|
        score_log_proob_given_motif = torch.logsumexp(score, dim = 0) - torch.log(torch.tensor(score.shape[0], device=self.device))
        #twisting score log_prob
        score_log_proob_given_motif = score_log_proob_given_motif
        return score_log_proob_given_motif        
    def systematic_resampling(self, ts, features, motif_index_mask):
        #TODO: implement systematic resampling
        return 0
    def gaussian_reparameterization(self, mean, var, reference_shape):
        #mean is in the shape of (batch_size, num_particles, sequence_length, 3)
        #var is in the shape of (batch_size, num_particles, sequence_length, 3)
        #reference_shape is in the shape of (batch_size, num_particles, sequence_length, 3)
        return mean + var.sqrt() * torch.randn_like(reference_shape)
    
    def check_gradient_accuracy(self, func, x, eps=1e-6):
        """
        Check gradient accuracy using finite differences.
        
        Args:
            func: Function that takes x as input and returns a scalar
            x: Input tensor
            eps: Small perturbation value
        
        Returns:
            max_diff: Maximum difference between analytical and numerical gradients
            avg_diff: Average difference between analytical and numerical gradients
        """
        # Get analytical gradient
        analytical_grad = torch.autograd.grad(func(x).sum(), x, create_graph=False)[0]
        
        # Compute numerical gradient
        numerical_grad = torch.zeros_like(x)
        for i in range(x.numel()):
            # Flatten x for easier indexing
            x_flat = x.flatten()
            
            # Forward difference
            x_flat[i] += eps
            forward = func(x_flat.reshape(x.shape))
            
            # Backward difference 
            x_flat[i] -= 2*eps
            backward = func(x_flat.reshape(x.shape))
            
            # Reset
            x_flat[i] += eps
            
            # Compute numerical gradient
            numerical_grad.flatten()[i] = (forward - backward).sum().item() / (2*eps)

        # Compare gradients
        diff = torch.abs(analytical_grad - numerical_grad)
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        
        return max_diff, avg_diff
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
        
        self.selected_motif_location_index = 10
        
        self.run = wandb.init(project="protein_design", name="test001")
        self.run.log({"num_samples": params['num_samples']})
        self.softmax_tracking = []
        
        number_of_particles = 10
        self.model.config.diffusion['n_timestep'] = 900
        
        params["num_particles"] = number_of_particles
        
        # Create features
        features = convert_np_features_to_tensor(
            batchify_np_features([
                self.create_np_features(params)
                for _ in range(params['num_samples']*params['num_particles'])
            ]),
            self.device
        )
        reshape_to_fit_diffusion = partial(self.reshape_trans_for_motif, batch_size = params['num_samples'], num_particles = params['num_particles'], reshape_for_diffusion_network = True)
        reshape_to_fit_twisting = partial(self.reshape_trans_for_motif, batch_size = params['num_samples'], num_particles = params['num_particles'], reshape_for_diffusion_network = False)
        

        # params['num_samples'] = params['num_particles']
        # features = convert_np_features_to_tensor(
        #     batchify_np_features([
        #         self.create_np_features(params)
        #         for _ in range(params['num_samples'])
        #     ]),
        #     self.device
        # )
        
        
        # Create frames
        trans = torch.randn_like(features['atom_positions'])
        trans.requires_grad = True
        rots = compute_frenet_frames(
            trans,
            features['chain_index'],
            features['residue_mask']
        )
        ts = T(rots.detach(), trans)

        # Define steps
        steps = reversed(np.arange(1, self.model.config.diffusion['n_timestep']))#reversed(np.arange(1, self.model.config.diffusion['n_timestep'] + 1))

        motif_target = self.load_motif_target()
        #use the first motif target for now
        motif_target = [motif_target[0]]
        motif_target = [torch.from_numpy(i).to(self.device) for i in motif_target]        
        motif_index_mask = self.generate_motif_index_mask(motif_target, trans)
        
        
        twisting_function = partial(self.twisting_function, motif_index_mask = motif_index_mask, motif_target = motif_target)
        
        
        timesteps = torch.Tensor([self.model.config.diffusion['n_timestep']] * params['num_samples']*params['num_particles']).int().to(self.device)
        z_pred = self.model.model(ts, timesteps, features)['z']
        
        
        # timesteps = torch.Tensor([self.model.config.diffusion['n_timestep']]).int().to(self.device)        
        # for key in features.keys():
        #     features[key] = features[key][0].unsqueeze(0)
        # z_pred_list = []
        # for i in range(ts.trans.shape[0]):
        #     ts_i = T(ts.rots[i].unsqueeze(0), ts.trans[i].unsqueeze(0))
        #     z_pred = self.model.model(ts_i, timesteps, features)['z']
        #     z_pred_list.append(z_pred)
        # z_pred = torch.cat(z_pred_list, dim = 0)
        
        
        x0_given_xt = (ts.trans - self.model.sqrt_one_minus_alphas_cumprod[self.model.config.diffusion['n_timestep']]*z_pred)/self.model.sqrt_alphas_cumprod[self.model.config.diffusion['n_timestep']]
        #initial twisting
        negative_log_prob_p_t_plus_1 =twisting_function(trans = reshape_to_fit_twisting(x0_given_xt), 
                                                        timesteps = self.model.config.diffusion['n_timestep'])
        w_t_plus_1 = torch.clamp(torch.exp(-torch.clamp(negative_log_prob_p_t_plus_1, min=-80)), min=1e-30)
        # grad_log_p_t_plus_1_given_x_t_plus_1 = torch.autograd.grad(negative_log_prob_p_t_plus_1, trans, grad_outputs = torch.ones_like(negative_log_prob_p_t_plus_1), 
        #                                                            retain_graph=False,
        #                                                            create_graph=False, allow_unused=True)[0]


        # Iterate
        for step in steps:
            # Define current diffusion timestep
            timesteps = torch.Tensor([step] * params['num_samples']*params['num_particles']).int().to(self.device)
            
            
            # Compute posterior, diffusion t-1
            # w_z = (1. - self.model.alphas[timesteps]) / self.model.sqrt_one_minus_alphas_cumprod[timesteps]
            # trans_mean = (ts.trans - w_z.view(-1, 1, 1) * z_pred)
            # trans_mean = (1. / self.model.sqrt_alphas[timesteps]).view(-1, 1, 1) * trans_mean
            # trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)
            # trans_z = torch.randn_like(ts.trans)
            # trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
            # trans = trans_mean + params['scale'] * trans_sigma * trans_z
            # trans = trans * features['residue_mask'].unsqueeze(-1)
            # trans = trans_mean
            
            
            #In the field of t+1
            #grad_log_p_t_plus_1_given_x_t_plus_1_list = []
            
            # for i in range(params['num_samples']*params['num_particles']):
            #     negative_log_prob_p_t_plus_1_i = negative_log_prob_p_t_plus_1[0,i]
            #     temp = torch.autograd.grad(negative_log_prob_p_t_plus_1_i, ts.trans[i], allow_unused=True, retain_graph=True)[0]
            #     grad_log_p_t_plus_1_given_x_t_plus_1_list.append(temp)
            #grad_log_p_t_plus_1_given_x_t_plus_1 = torch.autograd.grad(negative_log_prob_p_t_plus_1, ts.trans, allow_unused=False)[0]
            
            # for i in range(params['num_samples']*params['num_particles']):
            #     if i == params['num_samples']*params['num_particles'] - 1:
            #         negative_log_prob_p_t_plus_1[0,i].backward()
            #     else:   
            #         negative_log_prob_p_t_plus_1[0,i].backward(retain_graph=True)
                
            #negative_log_prob_p_t_plus_1.sum().backward()
            
            #take gradient of negative_log_prob_p_t_plus_1 with respect to ts.trans. For each particle, the gradient is different. 
            
            # def twisting_wrapper(x):
            #     rots = compute_frenet_frames(
            #         x,
            #         features['chain_index'], 
            #         features['residue_mask']
            #     )
            #     ts_temp = T(rots.detach(), x)
            #     x0_given_xt = (ts_temp.trans - self.model.sqrt_one_minus_alphas_cumprod[step]*z_pred)/self.model.sqrt_alphas_cumprod[step]
            #     return twisting_function(trans=reshape_to_fit_twisting(x0_given_xt), timesteps=step)

            # # Check gradient accuracy
            # max_diff, avg_diff = self.check_gradient_accuracy(twisting_wrapper, ts.trans)
            grad_log_p_t_plus_1_given_x_t_plus_1 = torch.autograd.grad(negative_log_prob_p_t_plus_1.sum(), ts.trans, allow_unused=False)[0]
            
            
            
            #grad_log_p_t_plus_1_given_x_t_plus_1 = ts.trans.grad
            #stabilize the gradient 
            #grad_log_p_t_plus_1_given_x_t_plus_1 = torch.clamp(grad_log_p_t_plus_1_given_x_t_plus_1, min=-100, max=100)
            
            twisting_score = -z_pred/self.model.sqrt_one_minus_alphas_cumprod[step+1] + grad_log_p_t_plus_1_given_x_t_plus_1
            
                
            gaussian_mean = 1/self.model.sqrt_alphas[step+1]*ts.trans + self.model.betas[step+1]/self.model.sqrt_alphas[step+1]*twisting_score
            gaussian_var = self.model.betas[step+1]
            trans_x_t = self.gaussian_reparameterization(gaussian_mean, gaussian_var, ts.trans).detach()
            
            rots = compute_frenet_frames(
                    trans_x_t,
                    features['chain_index'],
                    features['residue_mask']
                )
            ts = T(rots.detach(), trans_x_t.detach())
            ts.trans.requires_grad = True
            z_pred = self.model.model(ts, timesteps, features)['z']
            
            posterior_mean_coef1 = 1/self.model.sqrt_alphas_cumprod[step]
            posterior_mean_coef2 = (self.model.sqrt_one_minus_alphas_cumprod[step]/self.model.sqrt_alphas_cumprod[step])

            x0_given_xt = trans_x_t * posterior_mean_coef1 - z_pred * posterior_mean_coef2
            
            #x0_given_xt = (trans_x_t - self.model.sqrt_one_minus_alphas_cumprod[step]*z_pred)/self.model.sqrt_alphas_cumprod[step]
            negative_log_prob_p_t_plus_1 = twisting_function(trans = reshape_to_fit_twisting(x0_given_xt), 
                                                timesteps = step)
            wk = None
            
            # # Compute twisting
            # # TODO: add variance in the twisting function. Done=
            # #twisting_score = (self.model.sqrt_alphas_cumprod[step+1]*trans-ts.trans)/self.model.sqrt_one_minus_alphas_cumprod[step+1]
            
            # #twisting_score = -z_pred/self.model.sqrt_one_minus_alphas_cumprod[step]
            
            # #calculated predicted x0 given xt
            # x0_given_xt = (ts.trans - self.model.sqrt_one_minus_alphas_cumprod[step]*z_pred)/self.model.sqrt_alphas_cumprod[step]
            
            # twisting_score = (self.model.sqrt_alphas_cumprod[step]*x0_given_xt - ts.trans)/self.model.one_minus_alphas_cumprod[step]

            # twisting_score = twisting_score + grad_log_p_t_plus_1_given_x_t_plus_1
            
            # #Line 7
            # gaussian_mean = 1/self.model.sqrt_alphas[step]*ts.trans + self.model.betas[step]/self.model.sqrt_alphas[step]*twisting_score
            # gaussian_var = self.model.betas[step]
            # trans_x_t = self.gaussian_reparameterization(gaussian_mean, gaussian_var, ts.trans)
            
            # rots = compute_frenet_frames(
            #         trans_x_t,
            #         features['chain_index'],
            #         features['residue_mask']
            #     )
            # ts = T(rots, trans_x_t)
            # z_pred = self.model.model(ts, timesteps, features)['z']
            # twisting_score = -z_pred/self.model.sqrt_one_minus_alphas_cumprod[step]
            # #Line 8
            
            # negative_log_prob_p_t_plus_1 = twisting_function(trans = reshape_to_fit_twisting(trans_x_t), 
            #                                                  timesteps = step - 1)
            
            # #negative_log_prob_p_t_plus_1 = reshape_to_fit_diffusion(negative_log_prob_p_t_plus_1)
            # #Define the twisting score for the next step, for resampling purpose
            
            # #calculate the gradient one by one 
            # # grad_log_p_t_plus_1_given_x_t_plus_1 = torch.autograd.grad(negative_log_prob_p_t_plus_1.sum(), trans_x_t,grad_outputs = torch.ones_like(w_t_plus_1), 
            # #                                                            create_graph=False, allow_unused=True)[0]
            # grad_log_p_t_plus_1_given_x_t_plus_1 = torch.autograd.grad(negative_log_prob_p_t_plus_1.sum(), trans_x_t, 
            #                                                            create_graph=False, allow_unused=True)[0]
            
            
            # wk = None
            #TODO: implement the resampling function
            #Leave it for now
            # trans = trans_x_t
            
            # # Sample
            # if step == 1:

            #     # Compute rotations
            #     rots_mean = compute_frenet_frames(
            #         trans,
            #         features['chain_index'],
            #         features['residue_mask']
            #     )

            #     # Compute frames
            #     ts = T(rots_mean.detach(), trans.detach())

            # else:

            #     # # Compute translations
            #     # trans_z = torch.randn_like(ts.trans)
            #     # trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
            #     # trans = trans_mean + params['scale'] * trans_sigma * trans_z
            #     # trans = trans * features['residue_mask'].unsqueeze(-1)

            #     # Compute rotations
            #     rots = compute_frenet_frames(
            #         trans,
            #         features['chain_index'],
            #         features['residue_mask']
            #     )

            #     # Compute frames
            #     #ts = T(rots.detach(), trans.detach())
            #     ts = T(rots, trans)

        # Postprocess
        features['atom_positions'] = ts.trans.detach().cpu()
        np_features = convert_tensor_features_to_numpy(features)
        list_np_features = debatchify_np_features(np_features)
        #end of the run
        self.run.finish()
        
        #write a function that find the max position of a softmax score
        def find_max_position(softmax_score):
            self.softmax_tracking = np.array(self.softmax_tracking)
            return np.argmax(softmax_score, axis = 1)
        
        #find the max position of the softmax score
        max_position = find_max_position(self.softmax_tracking)
        print(f"The max position of the softmax score is {max_position}")
        
        return list_np_features