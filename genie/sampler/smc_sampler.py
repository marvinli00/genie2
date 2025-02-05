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
import os
from genie.sampler import BaseSampler
from genie.sampler.unconditional import UnconditionalSampler
from genie.sampler.twisted_diffusion.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights
from genie.sampler.twisted_diffusion.feynman_kac_pf import smc_FK


class SMCSampler(UnconditionalSampler):
    """
    SMCSampler for protein structure generation using Sequential Monte Carlo,
    now integrating smc_FK from twisted_diffusion_sampler.
    """
    
    def setup(self):
        self.add_required_parameter('length')

    def on_sample_start(self, params):
        """Set up output directories if necessary."""
        print(f"Starting SMC sampling with params: {params}")
        pdbs_dir = os.path.join(params['outdir'], 'pdbs')
        if not os.path.exists(pdbs_dir):
            os.makedirs(pdbs_dir)

    def create_np_features(self, params):
        """Creates a feature dictionary in numpy."""
        return create_empty_np_features([params['length']])
        # num_residues = params.get('num_residues', 100) # Example, adjust as needed
        # np_features = {
        #     'aatype': np.zeros((num_residues, 20), dtype=np.float32), # All ALA
        #     'num_chains': np.array([1], dtype=np.int32),
        #     'num_residues': np.array([num_residues], dtype=np.int32),
        #     'num_residues_per_chain': np.array([num_residues], dtype=np.int32),
        #     'atom_positions': np.zeros((num_residues, 3), dtype=np.float32), # Initialized to zeros
        #     'residue_mask': np.ones(num_residues, dtype=np.float32),
        #     'residue_index': np.arange(num_residues, dtype=np.int32),
        #     'chain_index': np.zeros(num_residues, dtype=np.int32),
        #     'fixed_sequence_mask': np.zeros(num_residues, dtype=np.float32),
        #     'fixed_structure_mask': np.zeros((num_residues, num_residues), dtype=np.float32),
        #     'fixed_group': np.zeros(num_residues, dtype=np.int32),
        #     'interface_mask': np.zeros(num_residues, dtype=np.float32),
        # }
        # return np_features

    def on_sample_end(self, params, list_np_features):
        # """Save generation outputs."""
        # print(f"SMC sampling finished, saving outputs to {params['outdir']}.")
        # for i, np_feature in enumerate(list_np_features):
        #     filename = f"{params['outdir']}/{params['prefix']}_sample_{params['offset'] + i}.npz"
        #     print(f"Saving sample {i} to {filename}")
        #     # np.savez_compressed(filename, **np_feature) # Uncomment to save if needed
        for i, np_features in enumerate(list_np_features):
            name = '{}_{}'.format(params['prefix'], params['offset'] + i)
            output_pdb_filepath = os.path.join(
                params['outdir'], 'pdbs', 
                '{}.pdb'.format(name)
            )
            save_np_features_to_pdb(np_features, output_pdb_filepath)

    def get_system(self): # Kept for potential future use, but G is defined inline in _sample now.
        """Returns log_prob_target function. May be used later."""
        def log_prob_target(particle_features):
            atom_positions = particle_features['atom_positions']
            residue_mask = particle_features['residue_mask']
            score = -torch.sum(atom_positions[residue_mask.bool()]**2)
            return score.unsqueeze(0)
        return log_prob_target, None

    def get_proposal_fn(self, params): # Kept for potential future use, but M is defined inline in _sample now.
        """Returns proposal function. May be used later."""
        model = self.model
        device = self.device
        scale = params['scale']

        def proposal_function(current_particle_features_list):
            proposed_particles_features_list = []
            for current_np_features in current_particle_features_list:
                features = convert_np_features_to_tensor(
                    batchify_np_features([current_np_features]), device
                )
                trans = features['atom_positions']
                rots = compute_frenet_frames(
                    trans,
                    features['chain_index'],
                    features['residue_mask']
                )
                ts = T(rots, trans)
                step = model.config.diffusion['n_timestep']
                timesteps = torch.Tensor([step] * 1).int().to(device)
                with torch.no_grad():
                    z_pred = model.model(ts, timesteps, features)['z']
                w_z = (1. - model.alphas[timesteps]) / model.sqrt_one_minus_alphas_cumprod[timesteps]
                trans_mean = (1. / model.sqrt_alphas[timesteps]).view(-1, 1, 1) * (ts.trans - w_z.view(-1, 1, 1) * z_pred)
                trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)
                trans = trans_mean
                trans = trans * features['residue_mask'].unsqueeze(-1)
                rots = compute_frenet_frames(
                    trans,
                    features['chain_index'],
                    features['residue_mask']
                )
                ts = T(rots.detach(), trans.detach())
                features['atom_positions'] = ts.trans.detach().cpu()
                proposed_np_features = convert_tensor_features_to_numpy(features)
                proposed_particles_features_list.append(debatchify_np_features(proposed_np_features)[0])
            return proposed_particles_features_list
        return proposal_function

    def get_prior_sample(self, params):
        """Samples initial particles."""
        num_particles = params['num_samples']
        initial_particles_features_list = []
        for _ in range(num_particles):
            np_features = self.create_np_features(params)
            np_features['atom_positions'] = np.random.randn(*np_features['atom_positions'].shape).astype(np.float32)
            initial_particles_features_list.append(np_features)
        return initial_particles_features_list

    def particle_to_features(self, particle):
        """Converts particle to feature dictionary."""
        return particle

    def _sample(self, params):
        """Sampling structures using smc_FK."""
        num_particles = params['num_samples']
        num_smc_steps = params.get('num_smc_steps', 10) # Example parameter
        resample_strategy = params.get('resample_strategy', 'systematic') # Example strategy
        ess_threshold = params.get('ess_threshold', 0.5) # Example ESS threshold

        initial_particles = self.get_prior_sample(params)
        particles = initial_particles

        # --- Define M (Transition Kernel / Proposal) ---
        def M_fn(t, xtp1, extra_vals, P):
            """Proposal function adapted to smc_FK's M format."""
            if t == num_smc_steps: # Initial step, sample from prior. smc_FK is reverse time, T is the start.
                particle_features_list = self.get_prior_sample(params) # Sample new particles from prior
            else: # Subsequent steps, propose from previous particles
                particle_features_list = [self.particle_to_features(p) for p in xtp1] # xtp1 is list of particles from previous step
                particle_features_list = self.get_proposal_fn(params)(particle_features_list) # Apply proposal
            return particle_features_list, extra_vals # extra_vals is not used here in proposal

        # --- Define G (Potential function / Reweighting) ---
        def G_fn(t, xtp1, xt, extra_vals):
            """Potential function adapted to smc_FK's G format."""
            log_weights = []
            for particle in xt: # xt is list of current particles from M_fn
                particle_tensor = convert_np_features_to_tensor(batchify_np_features([particle]), self.device)
                log_prob = self.get_system()[0](particle_tensor) # Using placeholder target from get_system
                log_weights.append(log_prob)
            log_weights_tensor = torch.cat(log_weights).squeeze(1) # Shape [P]
            return log_weights_tensor, extra_vals # extra_vals is not used here in potential

        # Convert initial particles to the format expected by smc_FK (numpy arrays, or lists)
        initial_particles_np = [convert_tensor_features_to_numpy(convert_np_features_to_tensor(batchify_np_features([p]), self.device)) for p in particles]
        initial_particles_np = [debatchify_np_features(p)[0] for p in initial_particles_np]


        # Run smc_FK
        final_particles, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace = \
            smc_FK(M=M_fn, G=G_fn, resample_strategy=resample_strategy, T=num_smc_steps, P=num_particles,
                   ess_threshold=ess_threshold, verbose=True, log_xt_trace=True, extra_vals={})

        # Convert final particles back to feature dictionaries and list_np_features
        list_np_features = [self.particle_to_features(p) for p in final_particles]

        return list_np_features