import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # [1, L, L]
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    token_weights = attn_weights.mean(dim=1).squeeze(0)  # [L]
    return token_weights.detach()

def aggregate_spans_for_model(hidden_states, layer_weights, attention_mask, offsets_mapping, spans_offsets, entropy_weights=None):
    
    device = hidden_states.device
    B_size, SeqLen, D = hidden_states.shape
    
    span_tensors = [
        torch.tensor(s, dtype=torch.long, device=device) if len(s) > 0 
        else torch.empty((0, 2), dtype=torch.long, device=device)
        for s in spans_offsets
    ]
    padded_spans = pad_sequence(span_tensors, batch_first=True, padding_value=0)
    
    if padded_spans.numel() == 0 or padded_spans.size(1) == 0:
        return None, None, None, None
        
    max_spans = padded_spans.size(1)
    padded_span_starts = padded_spans[:, :, 0]
    padded_span_ends = padded_spans[:, :, 1]
    
    lengths = torch.tensor([len(s) for s in spans_offsets], device=device) 
    col_indices = torch.arange(max_spans, device=device).unsqueeze(0)
    valid_span_mask = col_indices < lengths.unsqueeze(1)
    
    current_offsets = offsets_mapping[:, :SeqLen, :] if offsets_mapping.shape[1] != SeqLen else offsets_mapping
    offsets_start = current_offsets[..., 0].unsqueeze(2) 
    offsets_end = current_offsets[..., 1].unsqueeze(2)   
    
    span_starts_exp = padded_span_starts.unsqueeze(1)    
    span_ends_exp = padded_span_ends.unsqueeze(1)        
    
    token_in_span_map = (offsets_start + 1 >= span_starts_exp) & (offsets_end <= span_ends_exp)
    token_in_span_map = token_in_span_map & attention_mask.unsqueeze(2).bool() 
    
    A = token_in_span_map.transpose(1, 2).float() 
    
    weighted_hidden = hidden_states * layer_weights.unsqueeze(-1) 
    span_sum = torch.bmm(A, weighted_hidden)               
    weight_sum = torch.bmm(A, layer_weights.unsqueeze(-1)).squeeze(-1) 
    
    if entropy_weights is not None:
        ent_weight_sum = torch.bmm(A, entropy_weights.unsqueeze(-1)).squeeze(-1)
        span_lengths = A.sum(dim=-1).clamp(min=1e-5) 
        ent_weight_mean = ent_weight_sum / span_lengths 
        final_ent_weight = ent_weight_mean
    else:
        final_ent_weight = weight_sum
    
    span_repr = span_sum / weight_sum.unsqueeze(-1).clamp(min=1e-5)      
    
    return span_repr, weight_sum, final_ent_weight, valid_span_mask


def compute_hidden_span_loss(projector, s_span_repr, t_span_repr, valid_span_mask, w_sum):
    device = s_span_repr.device
    B_size, Max_Spans = valid_span_mask.shape
    
    # s_span_proj = projector(s_span_repr) 
    
    # valid_s = s_span_proj[valid_span_mask] # (N_valid, D)
    valid_s = s_span_repr[valid_span_mask] # (N_valid, D)
    valid_t = t_span_repr[valid_span_mask] # (N_valid, D)
    valid_w = w_sum[valid_span_mask]       # (N_valid)
    
    if valid_s.size(0) == 0:
        return torch.tensor(0.0, device=device)
        
    batch_indices = torch.arange(B_size, device=device).unsqueeze(1).expand(-1, Max_Spans)
    valid_batch_ids = batch_indices[valid_span_mask] 
    
    S_normalized = F.normalize(valid_s, p=2, dim=-1)
    T_normalized = F.normalize(valid_t, p=2, dim=-1)
    
    S_sim_matrix = S_normalized @ S_normalized.T
    T_sim_matrix = T_normalized @ T_normalized.T
    
    Same_Batch_Mask = (valid_batch_ids.unsqueeze(1) == valid_batch_ids.unsqueeze(0))
    Not_Self_Mask = ~torch.eye(valid_s.size(0), dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask
    
    S_intra_batch_similarities_flat = torch.masked_select(S_sim_matrix, Final_Mask)
    T_intra_batch_similarities_flat = torch.masked_select(T_sim_matrix, Final_Mask)
    
    Pair_Weights_Matrix = valid_w.unsqueeze(1) * valid_w.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)
    
    span_rel_loss = F.mse_loss(S_intra_batch_similarities_flat, T_intra_batch_similarities_flat, reduction='none')
    span_rel_loss = (span_rel_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)
    
    
    return span_rel_loss


def get_span_loss(projectors, s_att_mask, t_att_mask, s_hidden_states, t_hidden_states, 
                  s_offsets_mapping, t_offsets_mapping, spans_offsets, 
                  teacher_layer_mapping, student_layer_mapping, w_t_entropy=None):
    
    final_loss = 0.0
    for i, (s_idx, t_idx, projector) in enumerate(zip(student_layer_mapping, teacher_layer_mapping, projectors)):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]
        
        s_weights = compute_token_weights(s_hidden, s_att_mask) 
        t_weights = compute_token_weights(t_hidden, t_att_mask) 
        
        s_span_repr, _, _, valid_mask = aggregate_spans_for_model(s_hidden, s_weights, s_att_mask, s_offsets_mapping, spans_offsets)
        t_span_repr, t_weight_sum, t_ent_weight_sum, _ = aggregate_spans_for_model(t_hidden, t_weights, t_att_mask, t_offsets_mapping, spans_offsets, w_t_entropy)
        
        if s_span_repr is None or t_span_repr is None:
            continue
            
        w_sum = t_ent_weight_sum if w_t_entropy is not None else t_weight_sum
        span_loss = compute_hidden_span_loss(projector, s_span_repr, t_span_repr, valid_mask, w_sum)
        final_loss += span_loss

    return final_loss

def compute_overall_span_loss(projectors, s_att_mask, t_att_mask, s_logits, t_logits, 
                              s_hidden_states, t_hidden_states, 
                              s_offsets_mapping, t_offsets_mapping, 
                              spans_offsets, words_offsets, args):
    
    w_t_entropy = None
    if args.entropy_weight:
        t_probs = torch.softmax(t_logits.float().detach(), dim=-1)
        t_entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1)
        w_t_entropy = 1 - t_entropy / math.log(t_logits.size(-1))

    s_word_mapping = args.student_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    t_word_mapping = args.teacher_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_projectors = projectors[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    
    word_loss = get_span_loss(word_projectors, s_att_mask, t_att_mask, s_hidden_states, t_hidden_states, 
                              s_offsets_mapping, t_offsets_mapping, words_offsets, t_word_mapping, s_word_mapping, w_t_entropy)
    
    s_span_mapping = args.student_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    t_span_mapping = args.teacher_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_projectors = projectors[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    
    span_loss = get_span_loss(span_projectors, s_att_mask, t_att_mask, s_hidden_states, t_hidden_states, 
                              s_offsets_mapping, t_offsets_mapping, spans_offsets, t_span_mapping, s_span_mapping, w_t_entropy)
    
    overall_loss = (word_loss + span_loss) / len(args.student_layer_mapping)
    return overall_loss

def filter_overlapping_spans(spans):
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered = []
    words = []
    if not sorted_spans:
        return filtered

    current_span = sorted_spans[0]
    for next_span in sorted_spans[1:]:
        _, current_end, p = current_span
        _, next_end, _ = next_span
        if next_end <= current_end:
            continue
        filtered.append((current_span[0], current_span[1]))

        n_token = len(p)
        words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
        words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

        current_span = next_span
    filtered.append((current_span[0], current_span[1]))

    p = current_span[2]
    n_token = len(p)
    words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
    words.append((p[n_token - 1].idx, p[n_token-1].idx + len(p[n_token-1])))
    
    return filtered, words

def get_spans_offsets(texts, nlp, matcher):
    disabled_components = ["ner", "lemmatizer"]

    spans = []
    words = []

    for doc in nlp.pipe(texts, disable=disabled_components, n_process=4):
        spans_with_offsets = []
        
        vps = matcher(doc)
        for _, start, end in vps:
            vp = doc[start:end]
            spans_with_offsets.append((vp.start_char, vp.end_char, vp))
            
        ncs = doc.noun_chunks
        spans_with_offsets.extend([(nc.start_char, nc.end_char, nc) for nc in ncs])

        unique_spans, unique_words = filter_overlapping_spans(spans_with_offsets)
        spans.append(unique_spans)
        words.append(unique_words)
    
    return spans, words
