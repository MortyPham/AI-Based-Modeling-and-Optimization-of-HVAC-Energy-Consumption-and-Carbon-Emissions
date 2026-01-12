# training and evaluation functions
def train_epoch(model, loader, optimizer, A_hat):
    model.train()
    total_loss, total_bce, total_mse = 0.0, 0.0, 0.0

    for X_batch, yb_batch, yr_batch in loader:
        X_batch = X_batch.to(device)
        yb_batch = yb_batch.to(device)
        yr_batch = yr_batch.to(device)

        optimizer.zero_grad()
        logits, reg_out = model(X_batch, A_hat)

        loss_bce = bce_loss(logits, yb_batch)

        # Regression loss on active samples
        active_mask = (yb_batch > 0.5)
        if active_mask.sum() > 0:
            loss_reg = mse_loss(reg_out[active_mask], yr_batch[active_mask])
        else:
            loss_reg = torch.tensor(0.0, device=device)

        # Combine losses
        loss = loss_bce + 1.0 * loss_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_mse += loss_reg.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_bce / n_batches, total_mse / n_batches
