import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        gamma=2.0,
        beta=0.9999,
        lambda_=1.0,
        kappa=0.5,
        num_classes=10,
        reduction="mean",
        loss_type="focal",
    ):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.lambda_ = lambda_
        self.kappa = kappa
        self.num_classes = num_classes
        self.reduction = reduction

        self.loss_type = loss_type

    def calc_focal_loss(
        self, logits: torch.tensor, targets: torch.tensor
    ) -> torch.tensor:
        """
        Compute the focal loss using the focal weight and gamma parameters.
        Args:
            logits: torch.tensor, predicted logits
            targets: torch.tensor, binary targets
        Returns:
            torch.tensor, focal loss
        """
        probas = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy(probas, targets, reduction="none")
        pt = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = focal_weight * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss

    def calc_cb_focal_loss(
        self, logits: torch.tensor, targets: torch.tensor
    ) -> torch.Tensor:
        """
        Compute the class-balanced focal loss using the focal weight, gamma, and lambda parameters.
        Args:
            logits: torch.tensor, predicted logits
            targets: torch.tensor, binary targets
        Returns:
            torch.tensor, class-balanced focal loss
        """
        # Compute class probabilities
        p_ki = torch.sigmoid(logits)  # Probabilities for each class

        # Compute class counts
        batch_size, num_classes = targets.size()
        class_counts = targets.sum(dim=0).float()  # Number of instances per class

        # Calculate the effective number of samples for each class
        effective_num = 1.0 - self.beta**class_counts
        effective_num = torch.where(
            effective_num == 0, torch.ones_like(effective_num), effective_num
        )
        effective_num_weights = (1.0 - self.beta) / effective_num

        # Normalize the weights
        effective_num_weights /= effective_num_weights.sum()

        # Compute class balancing term rCB
        r_CB = (1.0 - self.beta) / (1.0 - self.beta**class_counts)
        r_CB = torch.where(
            class_counts == 0, torch.ones_like(r_CB), r_CB
        )  # Avoid division by zero

        # Compute focal loss components
        positive_loss = -((1 - p_ki) ** self.gamma) * torch.log(p_ki + 1e-6) * targets
        negative_loss = -(p_ki**self.gamma) * torch.log(1 - p_ki + 1e-6) * (1 - targets)

        # Compute the class-balanced focal loss
        class_balanced_focal_loss = r_CB * (positive_loss + negative_loss)

        return class_balanced_focal_loss

    def calc_db_loss(self, logits: torch.tensor, targets: torch.tensor) -> torch.Tensor:
        """
        Compute the double-binary focal loss using the focal weight, gamma, and lambda parameters.
        Args:
            logits: torch.tensor, predicted logits
            targets: torch.tensor, binary targets
        Returns:
            torch.tensor, double-binary focal loss
        """

        def smooth_rdb(r_db):
            """
            Smoothing function for rDB = alpha + sigma(beta * (rDB - mu))
            """
            mu = r_db.mean()
            return self.alpha + torch.sigmoid(self.beta * (r_db - mu))

        batch_size, num_classes = targets.size()
        class_counts = targets.sum(
            dim=0
        ).float()  # Convert to float for further calculations

        # Calculate P_C_i and P_I
        P_C_i = 1.0 / (num_classes * class_counts)
        P_I = 1.0 / (
            targets.sum(dim=0) + 1e-6
        )  # Add a small epsilon to avoid division by zero

        # Calculate rDB
        r_db = P_C_i / P_I
        r_db_hat = smooth_rdb(r_db)

        # Compute probabilities from logits
        q_ki = torch.sigmoid(logits)

        # Positive and negative losses
        positive_loss = (
            -r_db_hat * ((1 - q_ki) ** self.gamma) * torch.log(q_ki + 1e-6) * targets
        )
        negative_loss = (
            -r_db_hat
            * (1 / self.lambda_)
            * (q_ki**self.gamma)
            * torch.log(1 - q_ki + 1e-6)
            * (1 - targets)
        )

        # Compute class-specific bias vi
        class_prior = class_counts / batch_size
        b_hat = -torch.log(1.0 / class_prior - 1.0)
        vi = -self.kappa * b_hat

        # Apply class-specific bias and scale factor lambda for negative instances
        q_ki_positive = torch.sigmoid(logits - vi)
        q_ki_negative = torch.sigmoid(self.lambda_ * (logits - vi))

        # Final loss
        positive_loss_ntr = (
            -((1 - q_ki_positive) ** self.gamma)
            * torch.log(q_ki_positive + 1e-6)
            * targets
        )
        negative_loss_ntr = (
            -(1 / self.lambda_)
            * (q_ki_negative**self.gamma)
            * torch.log(1 - q_ki_negative + 1e-6)
            * (1 - targets)
        )

        # Combine positive and negative losses
        db_loss = positive_loss_ntr + negative_loss_ntr

        return db_loss

    def forward(self, logits, targets):

        probas = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy(probas, targets, reduction="none")

        if self.loss_type == "bce":
            loss = bce_loss
        elif self.loss_type == "focal":
            loss = self.calc_focal_loss(logits, targets)
        elif self.loss_type == "cb":
            loss = self.calc_cb_focal_loss(logits, targets)
        elif self.loss_type == "db":
            loss = self.calc_db_loss(logits, targets)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
