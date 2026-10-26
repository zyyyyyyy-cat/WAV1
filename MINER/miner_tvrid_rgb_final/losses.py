"""MINER contrastive loss (Equations 9-11 from the paper).

Bipartite cross-only contrastive loss. Computes symmetric cross-entropy
between query and gallery sets, with target mass assigned exclusively
to cross-set positives.

  Eq. 9:  L_q2g = -sum_i sum_j t_ij * log(softmax(sim_ij / tau))  [Q->G]
  Eq. 10: L_g2q = -sum_j sum_i t_ji * log(softmax(sim_ji / tau))  [G->Q]
  Eq. 11: L_total = 0.5 * (L_q2g + L_g2q)   [symmetric average]

where t_ij = 1/|P_i| if j is a positive for anchor i, 0 otherwise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPReIDLoss(nn.Module):
    """Bipartite cross-only contrastive loss (MINER, Equations 9-11).

    For each anchor in set Q, computes a softmax over all entries in set G,
    with uniform target mass over cross-set positives (same PID).
    Symmetric: averages Q->G and G->Q directions.

    Args:
        temperature: Initial temperature for logit scaling.
        learn_temperature: If True, temperature is a learnable parameter.
        label_smoothing: Label smoothing factor.
        min_temperature: Lower bound for learned temperature.
        max_temperature: Upper bound for learned temperature.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = False,
        label_smoothing: float = 0.0,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
    ):
        super().__init__()

        if learn_temperature:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
            )
        else:
            self.register_buffer(
                "logit_scale", torch.log(torch.tensor(1.0 / temperature))
            )
        self.learn_temperature = learn_temperature
        self.label_smoothing = label_smoothing

        # Clamping bounds: logit_scale = log(1/temp)
        self.logit_scale_min = torch.log(torch.tensor(1.0 / max_temperature)).item()
        self.logit_scale_max = torch.log(torch.tensor(1.0 / min_temperature)).item()

    def _get_logit_scale(self) -> torch.Tensor:
        """Return the clamped logit scale."""
        if self.learn_temperature:
            return torch.clamp(self.logit_scale, self.logit_scale_min, self.logit_scale_max)
        return self.logit_scale

    def get_temperature(self) -> torch.Tensor:
        """Return the effective temperature."""
        return 1.0 / torch.exp(self._get_logit_scale())

    @staticmethod
    def _label_matrix(labels_a: torch.Tensor, labels_b: torch.Tensor) -> torch.Tensor:
        return (labels_a[:, None] == labels_b[None, :]).float()

    def _apply_label_smoothing(self, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing == 0.0:
            return targets
        C = targets.size(1)
        return targets * (1 - self.label_smoothing) + self.label_smoothing / C

    def _direction_loss(
        self,
        logits_cross: torch.Tensor,
        labels_anchor: torch.Tensor,
        labels_cross: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy for one direction (Eq. 9 or 10)."""
        targets = self._label_matrix(labels_anchor, labels_cross)
        pos_counts = targets.sum(dim=1, keepdim=True).clamp(min=1)
        targets_norm = self._apply_label_smoothing(targets / pos_counts)
        log_probs = F.log_softmax(logits_cross, dim=1)
        return -(targets_norm * log_probs).sum(dim=1).mean()

    def forward(
        self,
        feats_q: torch.Tensor,
        feats_g: torch.Tensor,
        labels_q: torch.Tensor,
        labels_g: torch.Tensor,
    ) -> dict:
        """Compute symmetric bipartite cross-only contrastive loss (Eq. 11).

        Args:
            feats_q: (Nq, D) query features.
            feats_g: (Ng, D) gallery features.
            labels_q: (Nq,) query identity labels.
            labels_g: (Ng,) gallery identity labels.

        Returns:
            Dict with 'loss' (scalar) and 'temperature' (detached scalar).
        """
        feats_q = F.normalize(feats_q, p=2, dim=1)
        feats_g = F.normalize(feats_g, p=2, dim=1)

        scale = torch.exp(self._get_logit_scale())
        logits_qg = (feats_q @ feats_g.t()) * scale

        # Early exit if no cross-set positives
        if self._label_matrix(labels_q, labels_g).sum() == 0:
            loss = torch.tensor(0.0, device=feats_q.device, requires_grad=True)
            return {"loss": loss, "temperature": self.get_temperature().detach()}

        # Symmetric loss: Q->G (Eq. 9) + G->Q (Eq. 10), averaged (Eq. 11)
        loss_q2g = self._direction_loss(logits_qg, labels_q, labels_g)
        loss_g2q = self._direction_loss(logits_qg.t(), labels_g, labels_q)
        loss = 0.5 * (loss_q2g + loss_g2q)

        return {"loss": loss, "temperature": self.get_temperature().detach()}
