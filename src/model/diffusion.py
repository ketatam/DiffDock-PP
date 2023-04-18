"""
This file is a pain to read so I'm using it with minimal
modifications.

2022.11.08
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import BatchNorm
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean

from geom_utils import NoiseSchedule, score_norm
from .utils import _init


class AtomEmbedding(nn.Module):
    """
        Embeddings for atom identity only (as of now)
    """
    def __init__(self, args, num_atom):
        super(AtomEmbedding, self).__init__()
        # add 1 for padding
        self.atom_ebd = nn.Embedding(num_atom+1, args.ns)
        self.sigma_ebd = nn.Linear(args.sigma_embed_dim, args.ns)
        # LM embedding (ESM2)
        self.lm_ebd_dim = args.lm_embed_dim
        if self.lm_ebd_dim > 0:
            self.lm_ebd = nn.Linear(self.lm_ebd_dim + args.ns, args.ns)

        _init(self)

    def forward(self, x):
        # atom sequence
        atom_ebd = self.atom_ebd(x[:,0:1].long()).squeeze()
        # sigma noise embedding
        sigma_start = 1 + self.lm_ebd_dim
        sigma_ebd = self.sigma_ebd(x[:,sigma_start:])
        # add together
        final_ebd = atom_ebd + sigma_ebd
        # consider LM embedding here
        if self.lm_ebd_dim > 0:
            x = torch.cat([final_ebd, x[:, 1:sigma_start]], dim=1)
            final_ebd = self.lm_ebd(x)
        return final_ebd


class TPCL(nn.Module):
    """
        Tensor product convolution layer
    """
    def __init__(self, args,
                 in_irreps, sh_irreps, out_irreps, n_edge_features,
                 residual=True, hidden_features=None, is_last_layer=False):
        super(TPCL, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_features, self.tensor_prod.weight_numel),
        )
        if not args.no_batch_norm and not is_last_layer:
            self.batch_norm = BatchNorm(out_irreps)
        else:
            self.batch_norm = None

        _init(self)

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        out_nodes=None,
        reduction="mean",
    ):
        """
        @param edge_index  [2, E]
        @param edge_sh  edge spherical harmonics
        """
        edge_src, edge_dst = edge_index
        tp = self.tensor_prod(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduction)

        if self.residual:
            new_shape = (0, out.shape[-1] - node_attr.shape[-1])
            padded = F.pad(node_attr, new_shape)
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, args, model_params,
        confidence_mode=False,
        confidence_dropout=0,
        confidence_no_batchnorm=True,
        num_confidence_outputs=1,
    ):
        super(TensorProductScoreModel, self).__init__()
        self.noise_schedule = NoiseSchedule(args)
        self.t_embedding = get_timestep_embedding(args)

        self.in_lig_edge_features = 4
        self.lig_max_radius = args.max_radius
        self.rec_max_radius = 30
        self.cross_max_dist = args.cross_max_dist
        self.dynamic_max_cross = args.dynamic_max_cross
        self.cross_cutoff_weight = args.cross_cutoff_weight
        self.cross_cutoff_bias = args.cross_cutoff_bias
        self.center_max_dist = 30
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)
        self.ns, self.nv = args.ns, args.nv
        ns, nv = self.ns, self.nv  # >>> sigh stupid notation lazy
        self.scale_by_sigma = args.scale_by_sigma
        self.no_torsion = args.no_torsion
        self.confidence_mode = confidence_mode
        self.num_conv = args.num_conv_layers

        num_atoms = model_params["num_residues"]
        self.atom_embed = AtomEmbedding(args, num_atoms)

        # protein edge encoder
        self.rec_edge_embed = nn.Sequential(
            nn.Linear(args.sigma_embed_dim + args.dist_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(ns, ns),
        )

        self.cross_edge_embed = nn.Sequential(
            nn.Linear(args.sigma_embed_dim + args.cross_dist_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(ns, ns),
        )

        # distance expansion
        self.lig_dist_exp = GaussianSmearing(
                0.0, self.lig_max_radius, args.dist_embed_dim)
        self.rec_dist_exp = GaussianSmearing(
                0.0, self.rec_max_radius, args.dist_embed_dim)
        self.cross_dist_exp = GaussianSmearing(
                0.0, self.cross_max_dist, args.cross_dist_embed_dim)

        if args.use_second_order_repr:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]

        intra_convs = []
        cross_convs = []
        for i in range(self.num_conv):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            params = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 3 * ns,
                "hidden_features": 3 * ns,
                "residual": False,
            }

            intra_convs.append(TPCL(args, **params))
            cross_convs.append(TPCL(args, **params))

        self.intra_convs = nn.ModuleList(intra_convs)
        self.cross_convs = nn.ModuleList(cross_convs)

        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if self.num_conv >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs),
            )
        else:
            # center of mass translation and rotation components
            self.center_dist_exp = GaussianSmearing(
                0.0, self.center_max_dist, args.dist_embed_dim
            )
            self.center_edge_embed = nn.Sequential(
                nn.Linear(args.dist_embed_dim + args.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(ns, ns),
            )

            self.final_conv = TPCL(
                args,
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f"2x1o + 2x1e",
                n_edge_features=2 * ns,
                residual=False,
                is_last_layer=True,
            )
            self.tr_final_layer = nn.Sequential(
                nn.Linear(1 + args.sigma_embed_dim, ns),
                nn.Dropout(args.dropout),
                nn.ReLU(),
                nn.Linear(ns, 1),
            )
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + args.sigma_embed_dim, ns),
                nn.Dropout(args.dropout),
                nn.ReLU(),
                nn.Linear(ns, 1),
            )

            if not self.no_torsion:
                # torsion angles components
                self.final_edge_embed = nn.Sequential(
                    nn.Linear(args.dist_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(ns, ns),
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TPCL(
                    args,
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f"{ns}x0o + {ns}x0e",
                    n_edge_features=3 * ns,
                    residual=False,
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(args.dropout),
                    nn.Linear(ns, 1, bias=False),
                )

        _init(self)

    def forward(self, batch):
        # get noise schedule
        tr_t = batch.complex_t["tr"]
        rot_t = batch.complex_t["rot"]
        tor_t = batch.complex_t["tor"]
        if not self.confidence_mode:
            tr_s, rot_s, tor_s = self.noise_schedule(tr_t, rot_t, tor_t)
        else:
            tr_s, rot_s, tor_s = tr_t, rot_t, tor_t

        # build ligand graph
        ligand_graph = self.build_rigid_graph(batch, "ligand")
        lig_node_attr = self.atom_embed(ligand_graph[0])
        lig_src, lig_dst = lig_edge_index = ligand_graph[1]
        lig_edge_attr = self.rec_edge_embed(ligand_graph[2])
        lig_edge_sh = ligand_graph[3]

        # build receptor graph
        receptor_graph = self.build_rigid_graph(batch, "receptor")
        rec_node_attr = self.atom_embed(receptor_graph[0])
        rec_src, rec_dst = rec_edge_index = receptor_graph[1]
        rec_edge_attr = self.rec_edge_embed(receptor_graph[2])
        rec_edge_sh = receptor_graph[3]

        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_s * self.cross_cutoff_weight + self.cross_cutoff_bias).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_dist
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.cross_conv_graph(
            batch, cross_cutoff
        )
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embed(cross_edge_attr)

        for idx in range(len(self.intra_convs)):
            # message passing within ligand graph (intra)
            lig_edge_attr_ = torch.cat([
                    lig_edge_attr,
                    lig_node_attr[lig_src, :self.ns],
                    lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.intra_convs[idx](
                lig_node_attr, lig_edge_index,
                lig_edge_attr_, lig_edge_sh)

            # message passing between two graphs (inter)
            rec2lig_edge_attr_ = torch.cat([
                    cross_edge_attr,
                    lig_node_attr[cross_lig, :self.ns],
                    rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.cross_convs[idx](
                rec_node_attr,
                cross_edge_index,
                rec2lig_edge_attr_,
                cross_edge_sh,
                out_nodes=lig_node_attr.shape[0])

            # message passing within receptor graph (intra)
            if idx != len(self.intra_convs) - 1:
                rec_edge_attr_ = torch.cat([
                        rec_edge_attr,
                        rec_node_attr[rec_src, :self.ns],
                        rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.intra_convs[idx](
                    rec_node_attr, rec_edge_index,
                    rec_edge_attr_, rec_edge_sh)

                lig2rec_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        lig_node_attr[cross_lig, : self.ns],
                        rec_node_attr[cross_rec, : self.ns]], -1)
                rec_inter_update = self.cross_convs[idx](
                    lig_node_attr,
                    torch.flip(cross_edge_index, dims=[0]),
                    lig2rec_edge_attr_,
                    cross_edge_sh,
                    out_nodes=rec_node_attr.shape[0])

            # padding original features
            lig_node_attr = F.pad(
                lig_node_attr,
                (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update
            if idx != len(self.intra_convs) - 1:
                rec_node_attr = F.pad(
                    rec_node_attr,
                    (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        # compute confidence score
        if self.confidence_mode:
            scalar_lig_attr = (
                torch.cat(
                    [lig_node_attr[:, : self.ns], lig_node_attr[:, -self.ns :]], dim=1
                )
                if self.num_conv >= 3
                else lig_node_attr[:, : self.ns]
            )
            # debug = scatter_mean(scalar_lig_attr, batch["ligand"].batch, dim=0)
            # print(f'debug.shape: {debug.shape}')
            # print(f'debug: {debug}')
            confidence = self.confidence_predictor(
                scatter_mean(scalar_lig_attr, batch["ligand"].batch, dim=0)
            ).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors
        (
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
        ) = self.center_conv_graph(batch)
        center_edge_attr = self.center_edge_embed(center_edge_attr)
        center_edge_attr = torch.cat(
            [center_edge_attr, lig_node_attr[center_edge_index[0], : self.ns]], -1
        )
        global_pred = self.final_conv(
            lig_node_attr,
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
            out_nodes=batch.num_graphs,
        )

        # ligand tr [3], ligand rot [3] ?
        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        batch.graph_sigma_emb = self.t_embedding(batch.complex_t["tr"])

        # fix the magnitude of tr and rot score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1)[:, None]
        tr_scale = self.tr_final_layer(
            torch.cat([tr_norm, batch.graph_sigma_emb], dim=1))
        tr_pred = (tr_pred / tr_norm) * tr_scale

        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1)[:, None]
        rot_scale = self.rot_final_layer(
            torch.cat([rot_norm, batch.graph_sigma_emb], dim=1))
        rot_pred = (rot_pred / rot_norm) * rot_scale

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_s.unsqueeze(1)
            rot_pred = rot_pred * score_norm(rot_s)[:, None]
            rot_pred = rot_pred.to(batch["ligand"].x.device)

        if self.no_torsion or batch["ligand"].edge_mask.sum() == 0:
            tor_pred = torch.empty(0, device=tr_pred.device)
            return tr_pred, rot_pred, tor_pred

        # >>> FIXED UP TO HERE

        # torsional components
        (
            tor_bonds,
            tor_edge_index,
            tor_edge_attr,
            tor_edge_sh,
        ) = self.bond_conv_graph(batch)
        tor_bond_vec = (
            batch["ligand"].pos[tor_bonds[1]] - batch["ligand"].pos[tor_bonds[0]]
        )
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics(
            "2e", tor_bond_vec, normalize=True, normalization="component"
        )
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat(
            [
                tor_edge_attr,
                lig_node_attr[tor_edge_index[1], : self.ns],
                tor_bond_attr[tor_edge_index[0], : self.ns],
            ],
            -1,
        )
        tor_pred = self.tor_bond_conv(
            lig_node_attr,
            tor_edge_index,
            tor_edge_attr,
            tor_edge_sh,
            out_nodes=batch["ligand"].edge_mask.sum(),
            reduction="mean",
        )
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_s[batch["ligand"].batch][
            batch["ligand", "ligand"].edge_index[0]
        ][batch["ligand"].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(
                torch.tensor(torus.score_norm(edge_sigma.cpu().numpy()))
                .float()
                .to(batch["ligand"].x.device)
            )

        return tr_pred, rot_pred, tor_pred

    def build_rigid_graph(self, batch, key):
        """
            Fixed rigid proteins.
            Adds noise information to existing embeddings.
        """
        batch[key].node_sigma_emb = self.t_embedding(
            batch[key].node_t["tr"]
        )  # tr rot and tor noise is all the same
        # if no ESM models, graph.x should still be flat
        if len(batch[key].x.shape) == 1:
            batch[key].x = batch[key].x[:,None]
        node_attr = torch.cat(
            [batch[key].x, batch[key].node_sigma_emb], 1
        )

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = batch[key, key].edge_index
        src, dst = edge_index
        edge_vec = batch[key].pos[dst.long()] - batch[key].pos[src.long()]

        edge_length_emb = self.rec_dist_exp(edge_vec.norm(dim=-1))
        edge_sigma_emb = batch[key].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec,
            normalize=True, normalization="component")

        return node_attr, edge_index, edge_attr, edge_sh

    def cross_conv_graph(self, batch, cross_dist_cutoff):
        """
            Builds the cross edges between ligand and receptor
        """
        if torch.is_tensor(cross_dist_cutoff):
            # different cutoff for every graph
            # (depends on the diffusion time)
            edge_index = radius(
                batch["receptor"].pos / cross_dist_cutoff[batch["receptor"].batch],
                batch["ligand"].pos / cross_dist_cutoff[batch["ligand"].batch],
                1,
                batch["receptor"].batch,
                batch["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            edge_index = radius(
                batch["receptor"].pos,
                batch["ligand"].pos,
                cross_dist_cutoff,
                batch["receptor"].batch,
                batch["ligand"].batch,
                max_num_neighbors=10000,
            )

        src, dst = edge_index
        edge_vec = batch["receptor"].pos[dst.long()] - batch["ligand"].pos[src.long()]

        edge_length_emb = self.cross_dist_exp(edge_vec.norm(dim=-1))
        edge_sigma_emb = batch["ligand"].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec,
            normalize=True, normalization="component")

        return edge_index, edge_attr, edge_sh

    def center_conv_graph(self, batch):
        """
            Builds the filter and edges for the convolution
            generating translational and rotational scores
        """
        edge_index = torch.cat(
            [
                batch["ligand"].batch.unsqueeze(0),
                torch.arange(len(batch["ligand"].batch))
                .to(batch["ligand"].x.device)
                .unsqueeze(0),
            ],
            dim=0,
        )

        center_pos, count = torch.zeros((batch.num_graphs, 3)).to(
            batch["ligand"].x.device
        ), torch.zeros((batch.num_graphs, 3)).to(batch["ligand"].x.device)
        center_pos.index_add_(
            0, index=batch["ligand"].batch, source=batch["ligand"].pos
        )
        center_pos = center_pos / torch.bincount(batch["ligand"].batch).unsqueeze(1)

        edge_vec = batch["ligand"].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_dist_exp(edge_vec.norm(dim=-1))
        edge_sigma_emb = batch["ligand"].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True,
            normalization="component"
        )
        return edge_index, edge_attr, edge_sh

    def bond_conv_graph(self, batch):
        """
            Builds the graph for the convolution between
            the center of the rotatable bonds and the neighbouring
            nodes
        """
        bonds = (
            batch["ligand", "ligand"].edge_index[:, batch["ligand"].edge_mask].long()
        )
        bond_pos = (batch["ligand"].pos[bonds[0]] + batch["ligand"].pos[bonds[1]]) / 2
        bond_batch = batch["ligand"].batch[bonds[0]]
        edge_index = radius(
            batch["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=batch["ligand"].batch,
            batch_y=bond_batch,
        )

        edge_vec = batch["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_dist_exp(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embed(edge_attr)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True,
            normalization="component"
        )

        return bonds, edge_index, edge_attr, edge_sh


class GaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))



def get_timestep_embedding(args):
    if args.embedding_type == "sinusoidal":
        emb_func = SinusoidalEmbedding(args)
    elif args.embedding_type == "fourier":
        emb_func = GaussianFourierProjection(args)
    else:
        raise NotImplemented
    return emb_func


class SinusoidalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.sigma_embed_dim
        self.scale = args.embedding_scale
        self.max_positions = 1e4

    def forward(self, x):
        """
        From https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        """
        assert len(x.shape) == 1
        x = self.scale * x

        half_dim = self.embed_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim,
                dtype=torch.float32,
                device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (x.shape[0], self.embed_dim)
        return emb


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.sigma_embed_dim
        self.scale = args.embedding_scale
        self.W = nn.Parameter(
            torch.randn(self.embed_dim // 2) * self.scale,
            requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)
        return emb

