import torch

@torch.no_grad()
def generate(
    model,
    start_ids,
    max_new_tokens,
    temperature=1.0
):
    model.eval()
    idx = start_ids

    for _ in range(max_new_tokens):
        # 🔑 keep only last max_len tokens
        idx_cond = idx[:, -model.max_len:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    return idx

