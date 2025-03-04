import os

from genie.sampler.base import BaseSampler
from genie.utils.feat_utils import (
    create_empty_np_features,
    save_np_features_to_pdb
)
import torch
import torch.nn as nn
import numpy as np

from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.feat_utils import (
    convert_np_features_to_tensor,
    convert_tensor_features_to_numpy,
    batchify_np_features,
    debatchify_np_features,
    save_np_features_to_pdb
)
from genie.sampler.unconditional import UnconditionalSampler
# from genie.sampler.twisted_diffusion.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights
# from genie.sampler.twisted_diffusion.feynman_k        ac_pf import smc_FK
import numpy as np  
import os
from Bio.PDB import PDBParser
from functools import partial
import wandb
import copy


class EMA(nn.Module):
    """
    Exponential Moving Average implementation for PyTorch.
    
    This class implements an EMA that can be applied to gradients or any tensor.
    """
    
    def __init__(self, decay=0.9, zero_initialization=False):
        """
        Initialize the EMA module.
        
        Args:
            decay (float): The decay rate for the EMA (default: 0.9)
            zero_initialization (bool): Whether to initialize the EMA value to zeros (default: False)
        """
        super(EMA, self).__init__()
        self.decay = decay
        self.zero_initialization = zero_initialization
        self.shadow = None
        self.initialized = False
    
    def forward(self, gradient_tensor):
        """
        Update the EMA with a new gradient tensor.
        
        Args:
            gradient_tensor (torch.Tensor): The gradient tensor to incorporate into the EMA
            
        Returns:
            torch.Tensor: The updated EMA value
        """
        # Initialize shadow value on first call
        if not self.initialized:
            if self.zero_initialization:
                self.shadow = torch.zeros_like(gradient_tensor)
            else:
                self.shadow = gradient_tensor.clone().detach()
            self.initialized = True
        
        # Update shadow with new gradient value
        self.shadow = self.shadow * self.decay + gradient_tensor * (1 - self.decay)
        
        return self.shadow
    
    def get_ema(self):
        """
        Get the current EMA value.
        
        Returns:
            torch.Tensor: The current EMA value, or None if not initialized
        """
        return self.shadow if self.initialized else None
    
    def reset(self):
        """
        Reset the EMA state.
        """
        self.shadow = None
        self.initialized = False

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
        path = "/hai/scratch/mli89/protein_design/genie2/6E6R_long_motif.pdb"
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
        """
        The p_title_function is a critical component that guides the diffusion process to incorporate 
        specific structural motifs into the generated protein.
        """
        score = 0
        
        #define a function to find mask == true range
        def find_mask_true_range(mask):
            """
            Find the range of True values in a boolean mask.
            
            Args:
                mask (torch.Tensor): Boolean mask tensor
            
            Returns:
                start (int): Start index of the True range
                end (int): End index of the True range
            """
            start = (mask == True).nonzero()[0].item()
            end = (mask == True).nonzero()[-1].item()+1
            return start, end
        
        for i in range(motif_index_mask.shape[1]):
            motif_index_mask_i = motif_index_mask[:,i,:,:]
            motif_target_i = motif_target[i]
            score_i = 0
            #for j in range(motif_index_mask_i.shape[0]):
            if True:
                motif_index_mask_ij = motif_index_mask_i[15,:,0]
                start, end = find_mask_true_range(motif_index_mask_ij)
                predicted_x0_M = trans[:,:,start:end,:]
                #center the predicted_x0_M
                predicted_x0_M = predicted_x0_M - predicted_x0_M.mean(dim=-2, keepdim=True)
                #calculate the score
                score_i = score_i - torch.sum((predicted_x0_M - motif_target_i[None,None,:]) ** 2, dim=(2,3)) / (2*self.model.one_minus_alphas_cumprod[timesteps])
                self.run.log({f"distances_of_motif_{i}": ((predicted_x0_M - motif_target_i[None,None,:])**2).mean()})
            score = score + score_i
        score_log_proob_given_motif = torch.logsumexp(score, dim=0) - torch.log(torch.tensor(score.shape[0], device=self.device))
        return score_log_proob_given_motif
    
    
    def twisting_function1(self, trans, motif_index_mask, motif_target, timesteps):
        """
        The twisting function is a critical component that guides the diffusion process to incorporate 
        specific structural motifs into the generated protein.
        
        Args:
            trans: Tensor of shape (batch_size, num_particles, sequence_length, 3) containing the 
                backbone coordinates (Ca atom positions) of the protein structures being generated.
            
            motif_index_mask: Tensor of shape (max_offsets, num_motifs, sequence_length, 3) representing
                            binary masks for all possible placements of the motifs in the protein structure.
                            Each entry in the first dimension corresponds to a different possible placement.
            
            motif_target: List of tensors, each of shape (motif_length, 3) containing the 3D coordinates
                        of the target motif structures to be incorporated.
                        
            timesteps: Current timestep in the diffusion process, used to scale the scores based on
                    noise level at this timestep.
                    
        Returns:
            score_log_proob_given_motif: Log probability score measuring how well the generated structures
                                        incorporate the target motifs, used to guide the sampling process.
        """
        # Initialize the total score to 0
        score = 0
        
        # Loop through each motif (supporting multiple motifs)
        for i in range(motif_index_mask.shape[1]):
            # Extract the mask for this specific motif, which contains all possible placements
            motif_index_mask_i = motif_index_mask[:,i,:,:]
            
            # For debugging, select only a specific subset of placements
            # Here, it selects only the 15th placement position for focused evaluation
            motif_index_mask_i = motif_index_mask_i[15:16,:,:]
            
            # Get the target coordinates for this specific motif
            motif_target_i = motif_target[i]
            
            # Extract the coordinates of the current structure at each possible motif placement
            # This uses Einstein summation for efficient tensor manipulation:
            # - 'o' indexes over possible offsets (placements)
            # - 'b' indexes over batch dimension
            # - 'k' indexes over particle dimension
            # - 'l' indexes over sequence length
            # - 'd' indexes over xyz coordinates
            ts_com_zero = torch.einsum('old,bkld->obkld', motif_index_mask_i, trans)
            
            # Select only the coordinates where the motif mask is True using masked_select
            ts_com_zero = ts_com_zero.masked_select(motif_index_mask_i[:,None,None,:])
            
            # Reshape to organize the selected coordinates properly
            ts_com_zero = ts_com_zero.reshape(motif_index_mask_i.shape[0], trans.shape[0], trans.shape[1], -1, 3)
            
            # Center the coordinates by subtracting the mean position
            # This ensures alignment based on the center of mass
            # The detach() prevents gradient flow through the centering operation
            ts_com_zero = ts_com_zero - ts_com_zero.mean(dim=-2, keepdim=True).detach()
            
            # Calculate the negative log probability (score) based on the squared distance between
            # the current structure and the target motif
            # The division by one_minus_alphas_cumprod[timesteps] scales the score based on noise level
            # This implements a Gaussian log probability: -||x-μ||²/(2σ²)
            score_i = -torch.sum((ts_com_zero - motif_target_i[None,None,None,:]) ** 2, dim=(3,4)) / (2*self.model.one_minus_alphas_cumprod[timesteps])

            # Log the mean squared distance for monitoring/debugging
            self.run.log({f"distances_of_motif_{i}": ((ts_com_zero - motif_target_i[None,None,None,:])**2).mean()})

            # Add this motif's score to the total score
            # Multiple motifs' scores are summed, giving equal weight to each motif
            score = score + score_i
            
        # Track the softmax distribution over possible motif placements over time
        # This is done in a no_grad context since it's just for monitoring
        with torch.no_grad():
            # Initialize a list to store softmax distributions at different timesteps
            softmax_scores_over_time = []

            # Calculate softmax of scores across placement options (first dimension)
            # This gives a probability distribution over different motif placements
            softmax_score = torch.softmax(score, dim=0)[:,0,0]
            
            # Store the softmax distribution for later analysis
            softmax_scores_over_time.append(softmax_score.cpu().numpy())
            self.softmax_tracking.append(softmax_score.cpu().numpy())
        
        # Compute the final score as a log-sum-exp over all possible motif placements
        # This implements: log(1/|M| * sum_m exp(score_m))
        # - logsumexp efficiently computes log(sum(exp(x))) in a numerically stable way
        # - Subtracting log(|M|) accounts for the 1/|M| term (uniform prior over placements)
        score_log_proob_given_motif = torch.logsumexp(score, dim=0) - torch.log(torch.tensor(score.shape[0], device=self.device))
        
        # Return the final log probability score
        return score_log_proob_given_motif

    def gaussian_reparameterization(self, mean, var, reference_shape):
        #mean is in the shape of (batch_size, num_particles, sequence_length, 3)
        #var is in the shape of (batch_size, num_particles, sequence_length, 3)
        #reference_shape is in the shape of (batch_size, num_particles, sequence_length, 3)
        return mean + var.sqrt() * torch.randn_like(reference_shape)
    
    def check_gradient_accuracy(self, func, x, eps=5e-7, key = 'z'):
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
        analytical_grad = torch.autograd.grad(func(x)[key].sum(), x.trans, create_graph=False, allow_unused=True)[0]
        # Compute numerical gradient
        with torch.no_grad():
            numerical_grad = torch.zeros_like(x.trans)
            for i in range(x.trans.numel()):
                # Flatten x for easier indexing
                x_flat = x.trans.flatten()
                
                # Forward difference
                x_flat[i] += eps
                
                input = T(x.rots, x_flat.reshape(x.trans.shape))
                
                forward = func(input)[key]
                
                # Backward difference 
                x_flat[i] -= 2*eps
                input = T(x.rots, x_flat.reshape(x.trans.shape))
                backward = func(input)[key]
                
                # Reset
                x_flat[i] += eps
                
                # Compute numerical gradient
                numerical_grad.flatten()[i] = (forward - backward).sum().item() / (2*eps)
                break
        # Compare gradients
        # diff = torch.abs(analytical_grad - numerical_grad)
        # max_diff = diff.max().item()
        # avg_diff = diff.mean().item()
        diff = torch.abs(analytical_grad.flatten()[0] - numerical_grad.flatten()[0])
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        print(f"The analytical gradient is {analytical_grad.flatten()[0]} and the numerical gradient is {numerical_grad.flatten()[0]}")
        print(f"The max difference is {max_diff} and the average difference is {avg_diff}")
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
                stores information on a generated structure.
        """
        
        # Set a specific motif location index for debugging/visualization purposes
        self.selected_motif_location_index = 10
        
        # Initialize Weights & Biases for experiment tracking and visualization
        self.run = wandb.init(project="protein_design", name="test001")
        self.run.log({"num_samples": params['num_samples']})
        # Initialize array to track softmax scores over time (for motif placement probabilities)
        self.softmax_tracking = []
        
        # Set the number of particles (samples) for the SMC algorithm
        # Here it's set to 1, but could be increased for better exploration
        number_of_particles = 10
        # Set the total number of diffusion timesteps to 900
        self.model.config.diffusion['n_timestep'] = 1000
        
        # Add number of particles to parameters dictionary
        params["num_particles"] = number_of_particles
        
        # Create feature tensors for each sample/particle
        # This initializes empty features for all samples that will be filled during sampling
        features = convert_np_features_to_tensor(
            batchify_np_features([
                self.create_np_features(params)
                for _ in range(params['num_samples']*params['num_particles'])
            ]),
            self.device
        )
        
        # Create partial functions for reshaping tensors between diffusion network format and twisting function format
        # These handle the tensor dimension transformations needed during sampling
        reshape_to_fit_diffusion = partial(self.reshape_trans_for_motif, batch_size = params['num_samples'], num_particles = params['num_particles'], reshape_for_diffusion_network = True)
        reshape_to_fit_twisting = partial(self.reshape_trans_for_motif, batch_size = params['num_samples'], num_particles = params['num_particles'], reshape_for_diffusion_network = False)
        
        # Create initial random coordinates (translation vectors) for the protein backbone
        # These are initialized with Gaussian noise
        trans = torch.randn_like(features['atom_positions'])
        # Enable gradient tracking for the translation vectors (needed for gradient-based guidance)
        trans.requires_grad = True
        
        # Compute Frenet frames from the random translations
        # Frenet frames provide local coordinate systems along the protein backbone
        rots = compute_frenet_frames(
            trans,
            features['chain_index'],
            features['residue_mask']
        )
        
        # Combine rotations and translations into a unified transformation object
        ts = T(rots.detach(), trans)

        # Define diffusion timesteps in reverse order (for denoising)
        # The range starts from n_timestep-1 and goes down to 1
        steps = reversed(np.arange(1, self.model.config.diffusion['n_timestep']))
        
        # Load the target motif structure from a PDB file
        motif_target = self.load_motif_target()
        # Use only the first motif target for simplicity
        motif_target = [motif_target[0]]
        # Convert motif target to PyTorch tensors and move to the appropriate device
        motif_target = [torch.from_numpy(i).to(self.device) for i in motif_target]
        
        # Generate masks for all possible placements of motifs in the protein structure
        motif_index_mask = self.generate_motif_index_mask(motif_target, trans)
        
        # Create a partial function for the twisting function with fixed parameters
        # The twisting function guides sampling toward structures containing the motifs
        twisting_function = partial(self.twisting_function, motif_index_mask = motif_index_mask, motif_target = motif_target)
        
        # Set initial timestep to the maximum diffusion step for all samples
        timesteps = torch.Tensor([self.model.config.diffusion['n_timestep']] * params['num_samples']*params['num_particles']).int().to(self.device)
        
        # Run the diffusion model to predict the noise component at the initial timestep
        z_pred = self.model.model(ts, timesteps, features)['z']
        
        # Estimate the clean structure (x0) given the current noisy structure (xt)
        # This uses the standard diffusion model formula to estimate the clean signal
        x0_given_xt = (ts.trans - self.model.sqrt_one_minus_alphas_cumprod[self.model.config.diffusion['n_timestep']]*z_pred)/self.model.sqrt_alphas_cumprod[self.model.config.diffusion['n_timestep']]
        
        # Compute the initial twisting score using the estimated clean structure
        # This evaluates how well the current structure incorporates the target motifs
        log_prob_p_t_plus_1 = twisting_function(
            trans = reshape_to_fit_twisting(x0_given_xt), 
            timesteps = self.model.config.diffusion['n_timestep']
        )
        
        # Convert negative log probabilities to weights, with clamping for numerical stability
        w_t_plus_1 = torch.clamp(torch.exp(-torch.clamp(log_prob_p_t_plus_1, min=-80)), min=1e-30)
        
        # Set model to evaluation mode
        self.model.eval()
        ema = EMA(decay=0.9, zero_initialization=True)

        

        
        # Create a deep copy of features for visualization/debugging purposes
        features_copy = copy.deepcopy(features)

        # Main denoising loop - iterates backward through diffusion timesteps
        for step in steps:
            # Set current timestep for all samples
            timesteps = torch.Tensor([step] * params['num_samples']*params['num_particles']).int().to(self.device)
            
            # Compute gradient of the twisting function with respect to the current structure
            # This gradient guides sampling toward structures containing the motifs
            grad_log_p_t_plus_1_given_x_t_plus_1 = torch.autograd.grad(
                log_prob_p_t_plus_1.reshape(-1).mean(), 
                ts.trans, 
                allow_unused=False
            )[0]
            
            # Combine diffusion model noise prediction with the twisting function gradient
            # This creates a "twisted" score for guiding the sampling process
            
            if step >= 980:
                alpha = 10
                grad_log_p_t_plus_1_given_x_t_plus_1 = grad_log_p_t_plus_1_given_x_t_plus_1*alpha/(alpha+(grad_log_p_t_plus_1_given_x_t_plus_1.norm(dim=[1,2])[:,None,None]))
                grad_log_p_t_plus_1_given_x_t_plus_1 = -grad_log_p_t_plus_1_given_x_t_plus_1
            else:
                alpha = 5
                grad_log_p_t_plus_1_given_x_t_plus_1 = grad_log_p_t_plus_1_given_x_t_plus_1*alpha/(alpha+(grad_log_p_t_plus_1_given_x_t_plus_1.norm(dim=[1,2])[:,None,None]))
                grad_log_p_t_plus_1_given_x_t_plus_1 = grad_log_p_t_plus_1_given_x_t_plus_1
            
            
            twisting_score = -z_pred/self.model.sqrt_one_minus_alphas_cumprod[step+1] + grad_log_p_t_plus_1_given_x_t_plus_1
            
            # Compute mean and variance for the posterior distribution p(x_t | x_{t+1})
            # These formulas are derived from the diffusion process equations
            gaussian_mean = 1/self.model.sqrt_alphas[step+1]*ts.trans + self.model.betas[step+1]/self.model.sqrt_alphas[step+1]*twisting_score
            gaussian_var = self.model.betas[step+1]
            
            # Sample the next structure using the reparameterization trick
            # This generates a new sample from the twisted posterior distribution
            trans_x_t = self.gaussian_reparameterization(gaussian_mean, gaussian_var, ts.trans).detach()
            
            # Recompute Frenet frames for the new structure
            rots = compute_frenet_frames(
                trans_x_t,
                features['chain_index'],
                features['residue_mask']
            )
            
            # Update the transformation object with new rotations and translations
            ts = T(rots.detach(), trans_x_t.detach())
            # Enable gradient tracking for the next iteration
            ts.trans.requires_grad = True
            
            # TODO 
            # Check gradient accuracy for debugging purposes
            # This compares numerical and analytical gradients to ensure correctness
            # fun_grad = partial(self.model.model, timesteps = timesteps, features = features)
            # temp = self.check_gradient_accuracy(fun_grad, ts, key = 'z', eps = 1e-7)
            # print(temp)
            # Run the diffusion model again to predict the noise component
            z_pred = self.model.model(ts, timesteps, features)['z']
            
            # Compute coefficients for estimating the clean structure
            posterior_mean_coef1 = 1/self.model.sqrt_alphas_cumprod[step]
            posterior_mean_coef2 = (self.model.sqrt_one_minus_alphas_cumprod[step]/self.model.sqrt_alphas_cumprod[step])

            # Estimate the clean structure from the current noisy structure
            x0_given_xt = trans_x_t * posterior_mean_coef1 - z_pred * posterior_mean_coef2
            
            # Every 50 steps, save the current estimated structure for visualization
            if step % 50 == 0:
                features_copy['atom_positions'] = x0_given_xt.detach().cpu().clone()
                np_features = convert_tensor_features_to_numpy(features_copy)
                list_np_features = debatchify_np_features(np_features)
                save_np_features_to_pdb(list_np_features[0], f"test/x0_predicted_test_{step}.pdb")
                
                
                features_copy['atom_positions'] = trans_x_t.detach().cpu().clone()
                np_features = convert_tensor_features_to_numpy(features_copy)
                list_np_features = debatchify_np_features(np_features)
                save_np_features_to_pdb(list_np_features[0], f"test/xt_predicted_test_{step}.pdb")
            
            # Compute the twisting function score for the new estimated clean structure
            # This evaluates how well the current structure incorporates the motifs
            log_prob_p_t_plus_1 = twisting_function(
                trans = reshape_to_fit_twisting(x0_given_xt), 
                timesteps = step
            )
            
            # Variable for importance weights (not used in this implementation)
            wk = None

        # Set the final atom positions in the feature dictionary
        features['atom_positions'] = ts.trans.detach().cpu()
        
        # Convert tensor features back to numpy
        np_features = convert_tensor_features_to_numpy(features)
        
        # Split batched features into individual structures
        list_np_features = debatchify_np_features(np_features)
        
        # Finish the Weights & Biases run
        self.run.finish()
        
        # Define a helper function to find the maximum position in softmax scores
        def find_max_position(softmax_score):
            self.softmax_tracking = np.array(self.softmax_tracking)
            return np.argmax(softmax_score, axis = 1)
        
        # Find the most likely motif placement positions
        max_position = find_max_position(self.softmax_tracking)
        print(f"The max position of the softmax score is {max_position}")
        
        # Return the list of generated protein structures
        return list_np_features