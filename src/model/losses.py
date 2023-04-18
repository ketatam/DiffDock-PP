import torch
import torch.nn as nn
import torch.nn.functional as F

from geom_utils import NoiseSchedule, score_norm


class DiffusionLoss(nn.Module):
    def __init__(self, args):
        super(DiffusionLoss, self).__init__()
        self.args = args
        self.tr_weight = args.tr_weight
        self.rot_weight = args.rot_weight
        self.tor_weight = args.tor_weight
        self.noise_schedule = NoiseSchedule(args)
        self.eps = 1e-5

    def forward(self, data, outputs,
                apply_mean=True, no_torsion=True):
        """
            @param (dict) outputs
            @param (torch_geometric.data.HeteroData) data
        """
        # extract outputs
        tr_pred = outputs["tr_pred"]
        rot_pred = outputs["rot_pred"]
        tor_pred = outputs["tor_pred"]
        device = tr_pred.device

        # gather t values
        complex_t = []
        for noise_type in ["tr", "rot", "tor"]:
            if torch.cuda.is_available() and self.args.num_gpu == 1:
                cur_t = data.complex_t[noise_type]
            else:
                cur_t = torch.cat([d.complex_t[noise_type] for d in data])
            complex_t.append(cur_t)

        # convert to sigmas
        tr_s, rot_s, tor_s = self.noise_schedule(*complex_t)
        mean_dims = (0, 1) if apply_mean else 1

        # translation component
        tr_score = (
            torch.cat([d.tr_score for d in data], dim=0)
            if device.type == "cuda" and self.args.num_gpu > 1
            else data.tr_score.cpu()
        )
        tr_s = tr_s.unsqueeze(-1).cpu()
        tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * tr_s**2).mean(dim=mean_dims)
        tr_base_loss = (tr_score**2 * tr_s**2).mean(dim=mean_dims).detach()

        # rotation component
        rot_score = (
            torch.cat([d.rot_score for d in data], dim=0)
            if device.type == "cuda" and self.args.num_gpu > 1
            else data.rot_score.cpu()
        )
        rot_score_norm = score_norm(rot_s.cpu()).unsqueeze(-1)
        rot_loss = (((rot_pred.cpu() - rot_score) /
            (rot_score_norm + self.eps)) ** 2).mean(
            dim=mean_dims
        )
        rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

        # torsion component
        if not no_torsion:
            edge_tor_s = torch.from_numpy(
                np.concatenate(
                    [d.tor_s_edge for d in data]
                    if device.type == "cuda"
                    else data.tor_s_edge
                )
            )
            tor_score = (
                torch.cat([d.tor_score for d in data], dim=0)
                if device.type == "cuda"
                else data.tor_score
            )
            tor_score_norm2 = torch.tensor(
                torus.score_norm(edge_tor_s.cpu().numpy())
            ).float()
            tor_loss = (tor_pred.cpu() - tor_score) ** 2 / tor_score_norm2
            tor_base_loss = ((tor_score**2 / tor_score_norm2)).detach()
            if apply_mean:
                tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(
                    1, dtype=torch.float
                ), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
            else:
                index = (
                    torch.cat(
                        [
                            torch.ones(d["ligand"].edge_mask.sum()) * i
                            for i, d in enumerate(data)
                        ]
                    ).long()
                    if device.type == "cuda"
                    else data["ligand"].batch[
                        data["ligand", "ligand"].edge_index[0][data["ligand"].edge_mask]
                    ]
                )
                num_graphs = len(data) if device.type == "cuda" else data.num_graphs
                t_l, t_b_l, c = (
                    torch.zeros(num_graphs),
                    torch.zeros(num_graphs),
                    torch.zeros(num_graphs),
                )
                c.index_add_(0, index, torch.ones(tor_loss.shape))
                c = c + 0.0001
                t_l.index_add_(0, index, tor_loss)
                t_b_l.index_add_(0, index, tor_base_loss)
                tor_loss, tor_base_loss = t_l / c, t_b_l / c
        else:
            if apply_mean:
                tor_loss = torch.zeros(1, dtype=torch.float)
                tor_base_loss = torch.zeros(1, dtype=torch.float)
            else:
                tor_loss = torch.zeros(len(rot_loss),
                                       dtype=torch.float)
                tor_base_loss = torch.zeros(len(rot_loss),
                                            dtype=torch.float)

        # compile and re-weight losses
        loss = tr_loss * self.tr_weight
        loss = loss + rot_loss * self.rot_weight
        if not no_torsion:
            loss = loss + tor_loss * self.tor_weight

        losses = {
            "loss": loss,
            "tr_loss": tr_loss,
            "rot_loss": rot_loss,
            "tr_base_loss": tr_base_loss,
            "rot_base_loss": rot_base_loss,
        }
        if not no_torsion:
            losses.update({
                "tor_loss": tor_loss,
                "tor_base_loss": tor_base_loss
            })

        return losses

