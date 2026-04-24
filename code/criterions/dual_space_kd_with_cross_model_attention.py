import math
import torch
import torch.nn.functional as F
from .various_divergence import VariousDivergence
from utils import log_rank
import re
import spacy
from spacy.matcher import Matcher


def align_sequences(tea_seq, stu_seq, student_tokenizer, teacher_tokenizer):
    i, j = 0, 0
    t2s_align, s2t_align = [], []
    history_tea_seq, history_stu_seq = "", ""
    
    tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
    stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea_seq == history_stu_seq and (
            tea_seq[i] == stu_seq[j] or (
                tea_seq[i] == teacher_tokenizer.eos_token and \
                stu_seq[j] == student_tokenizer.eos_token
            )
        ):
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            t2s_align.append(i)
            s2t_align.append(j)
            i += 1
            j += 1
        elif len(history_tea_seq) > len(history_stu_seq):
            history_stu_seq += stu_seq[j]
            j += 1
        elif len(history_tea_seq) < len(history_stu_seq):
            history_tea_seq += tea_seq[i]
            i += 1
        else:
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            i += 1
            j += 1
            
    return t2s_align, s2t_align


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

def prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                     attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    B_size, SeqLen = attention_mask.shape

    max_spans = max(len(s) for s in spans_offsets)
    if max_spans == 0:
        print(f"No spans found in the batch.")
        return torch.tensor(0.0, device=device)

    # (B_size, max_spans)
    padded_span_starts = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_ends = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_mask = torch.zeros(B_size, max_spans, dtype=torch.bool, device=device)

    for i in range(B_size):
        num_spans_i = len(spans_offsets[i])
        if num_spans_i > 0:
            spans_i = torch.tensor(spans_offsets[i], device=device, dtype=torch.long)
            padded_span_starts[i, :num_spans_i] = spans_i[:, 0]
            padded_span_ends[i, :num_spans_i] = spans_i[:, 1]
            padded_span_mask[i, :num_spans_i] = True
    
    if offsets_mapping.shape[1] != SeqLen:
        current_offsets_mapping = offsets_mapping[:, :SeqLen, :]
    else:
        current_offsets_mapping = offsets_mapping

    # (B_size, SeqLen, 1)
    offsets_start_expanded = current_offsets_mapping[..., 0].unsqueeze(2).to(device)
    offsets_end_expanded = current_offsets_mapping[..., 1].unsqueeze(2).to(device)
    
    # (B_size, 1, max_spans)
    span_starts_expanded = padded_span_starts.unsqueeze(1)
    span_ends_expanded = padded_span_ends.unsqueeze(1)

    token_in_span_map = (offsets_start_expanded + 1 >= span_starts_expanded) & \
                        (offsets_end_expanded <= span_ends_expanded)

    attention_mask_expanded = attention_mask.unsqueeze(2).bool()
    span_mask_expanded = padded_span_mask.unsqueeze(1) 

    final_token_to_span_map = token_in_span_map & attention_mask_expanded & span_mask_expanded

    if not final_token_to_span_map.any():
        print(f"No valid tokens found for any spans in the batch.")
        return torch.tensor(0.0, device=device)

    nonzero_indices = final_token_to_span_map.nonzero(as_tuple=False)
    
    batch_indices = nonzero_indices[:, 0] # (T_total)
    token_indices = nonzero_indices[:, 1] # (T_total)
    local_span_indices = nonzero_indices[:, 2] # (T_total)

    All_Indices = batch_indices * SeqLen + token_indices

    global_span_ids_flat = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids_flat, return_inverse=True) # (T_total)
    Max_Spans = Span_IDs.max().item() + 1 # Tổng số span duy nhất

    Batch_ID_for_Spans = torch.empty(Max_Spans, device=device, dtype=torch.long)
    Batch_ID_for_Spans.scatter_(0, Span_IDs, batch_indices)

    def gather_layer_weights(layer_weights):
        B_size, SeqLen = attention_mask.shape
        num_layers = layer_weights.shape[0]
        layer_weights_flat = layer_weights.view(num_layers, B_size * SeqLen)
        token_weights_unnorm = layer_weights_flat[:, All_Indices].float()
        batch_indices_expanded = batch_indices.unsqueeze(0).expand(num_layers, -1)
        sample_weight_sums = torch.zeros(num_layers, B_size, device=device, dtype=torch.float)
        sample_weight_sums.scatter_add_(1, batch_indices_expanded, token_weights_unnorm)
        sample_weight_sums = sample_weight_sums.clamp(min=1e-5)
        sample_weight_sums_gathered = torch.gather(sample_weight_sums, 1, batch_indices_expanded)
        Token_Weights_all = token_weights_unnorm / sample_weight_sums_gathered

        return Token_Weights_all

    T_Token_Weights_all = gather_layer_weights(t_layer_weights)
    S_Token_Weights_all = gather_layer_weights(s_layer_weights)

    return All_Indices, T_Token_Weights_all, S_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans

def get_span_loss(projectors, attention_mask, s_hidden_states, t_hidden_states, 
                  offsets_mapping, spans_offsets, teacher_layer_mapping, student_layer_mapping):
    
    t_layer_weights = []
    s_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)  # (B, SeqLen)
        t_layer_weights.append(weights)
    for i in student_layer_mapping:
        weights = compute_token_weights(s_hidden_states[i], attention_mask)  # (B, SeqLen)
        s_layer_weights.append(weights)

    t_layer_weights = torch.stack(t_layer_weights)  # (num_layers, B, SeqLen)
    s_layer_weights = torch.stack(s_layer_weights)  # (num_layers, B, SeqLen)

    (All_Indices, T_Token_Weights_all, S_Token_Weights_all, 
     Span_IDs, Max_Spans, Batch_ID_for_Spans) =  prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                                                                  attention_mask, offsets_mapping, spans_offsets)
    final_loss = 0.0
    for i, (s_idx, t_idx, projector) in enumerate(zip(student_layer_mapping, teacher_layer_mapping, projectors)):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]
        span_loss = compute_hidden_span_loss(projector, s_hidden, t_hidden, All_Indices,
                                             S_Token_Weights_all[i], T_Token_Weights_all[i], 
                                             Span_IDs, Max_Spans, Batch_ID_for_Spans)
        final_loss += span_loss

    return final_loss

def get_token_loss(attention_mask, s_hidden_states, t_hidden_states, 
                   teacher_layer_mapping, student_layer_mapping):
    t_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)  # (B, SeqLen)
        t_layer_weights.append(weights)
    N = attention_mask.size(-1)
    final_loss = 0.0
    for i, (s_idx, t_idx) in enumerate(zip(student_layer_mapping, teacher_layer_mapping)):
        pair_weights = t_layer_weights[i].unsqueeze(2) * t_layer_weights[i].unsqueeze(1)
        mask = torch.eye(N, device=pair_weights.device).bool()  # (N, N)
        pair_weights[:, mask] = 0.0
        pair_weights = pair_weights / pair_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)

        s_tokens = F.normalize(s_hidden_states[s_idx], dim=-1, eps=1e-5)
        t_tokens = F.normalize(t_hidden_states[t_idx], dim=-1, eps=1e-5)
        student_scores = torch.matmul(s_tokens, s_tokens.transpose(-1, -2))
        teacher_scores = torch.matmul(t_tokens, t_tokens.transpose(-1, -2))
        span_loss = F.mse_loss(student_scores, teacher_scores, reduction='none')
        span_loss = (span_loss * pair_weights).sum() / pair_weights.sum()

        final_loss += span_loss

    final_loss = final_loss / len(student_layer_mapping)
    return final_loss

def compute_overall_span_loss(projectors, attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, words_offsets, args):
    
    s_word_mapping = args.student_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    t_word_mapping = args.teacher_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_projectors = projectors[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_loss = get_span_loss(word_projectors, attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, words_offsets, t_word_mapping, s_word_mapping)
    
    s_span_mapping = args.student_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    t_span_mapping = args.teacher_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_projectors = projectors[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_loss = get_span_loss(span_projectors, attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, t_span_mapping, s_span_mapping)
    
    overall_loss = (word_loss + span_loss) / len(args.student_layer_mapping)
    return overall_loss

def compute_hidden_span_loss(s_hidden_state, t_hidden_state, All_Indices, 
                             S_Token_Weights_all, T_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans):
    D_hidden_s = s_hidden_state.size(-1)
    D_hidden_t = t_hidden_state.size(-1)
    device = t_hidden_state.device

    T_Hidden_Flat = t_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_t)
    S_Hidden_Flat = s_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_s)

    T_span_all = T_Hidden_Flat[All_Indices] # (T_total, D_hidden_t)
    S_span_all = S_Hidden_Flat[All_Indices] # (T_total, D_hidden_s)
    
    T_Token_Weights_expanded = T_Token_Weights_all.unsqueeze(-1) 
    S_Token_Weights_expanded = S_Token_Weights_all.unsqueeze(-1)
    
    T_span_weighted = T_span_all * T_Token_Weights_expanded # (T_total, D_hidden_t)
    S_span_weighted = S_span_all * S_Token_Weights_expanded # (T_total, D_hidden_s)

    Span_IDs_expanded_t = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_t) 
    Span_IDs_expanded_s = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_s) 

    T_span_sum = torch.zeros(Max_Spans, D_hidden_t, device=device)
    S_span_sum = torch.zeros(Max_Spans, D_hidden_s, device=device)
    T_Weight_sum_1d = torch.zeros(Max_Spans, device=device)
    S_Weight_sum_1d = torch.zeros(Max_Spans, device=device)

    T_span_sum.scatter_add_(0, Span_IDs_expanded_t, T_span_weighted)
    S_span_sum.scatter_add_(0, Span_IDs_expanded_s, S_span_weighted)

    T_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Token_Weights_all) 
    T_Weight_sum = T_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)
    S_Weight_sum_1d.scatter_add_(0, Span_IDs, S_Token_Weights_all)
    S_Weight_sum = S_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)

    # Tính Trung bình (Mean)
    T_span_hidden_mean = T_span_sum / T_Weight_sum 
    S_span_hidden_mean = S_span_sum / S_Weight_sum

    S_normalized = F.normalize(S_span_hidden_mean, p=2, dim=-1)
    T_normalized = F.normalize(T_span_hidden_mean, p=2, dim=-1)
    S_Full_Sim_Matrix = S_normalized @ S_normalized.T
    T_Full_Sim_Matrix = T_normalized @ T_normalized.T

    Batch_IDs_col = Batch_ID_for_Spans.unsqueeze(1)
    Batch_IDs_row = Batch_ID_for_Spans.unsqueeze(0)
    Same_Batch_Mask = (Batch_IDs_col == Batch_IDs_row)
    Not_Self_Mask = ~torch.eye(Max_Spans, dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask

    S_intra_batch_similarities_flat = torch.masked_select(S_Full_Sim_Matrix, Final_Mask)
    T_intra_batch_similarities_flat = torch.masked_select(T_Full_Sim_Matrix, Final_Mask)

    Pair_Weights_Matrix = T_Weight_sum_1d.unsqueeze(1) * T_Weight_sum_1d.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)

    span_loss = F.mse_loss(S_intra_batch_similarities_flat, T_intra_batch_similarities_flat, reduction='none')
    span_loss = (span_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)

    return span_loss


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


class DualSpaceKDV2WithETA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        VERB_PHRASE_PATTERN = [
            {"POS": "AUX", "OP": "*"},
            {"POS": "ADV", "OP": "*"},
            {"POS": "VERB", "OP": "+"},
            {"POS": "ADV", "OP": "*"},
        ]

        self.matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])

    def forward(
        self, 
        distiller, 
        batch, 
        logging_output
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_model.eval()

        batch_input = batch["input_batch"]

        self.distiller = distiller
        outputs = model(**batch["input_batch"], output_hidden_states=True)
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, batch["label_batch"]["label"], reduction="sum"
        )[0] / batch["label_batch"]["loss_denom"]
        log["nll_loss"] = ce_loss

        if "op_input_batch" in batch:     # on-policy scenario
            outputs = model(**batch["op_input_batch"], output_hidden_states=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["op_teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_on_policy_dual_space_kd_loss_with_cma(
                outputs, teacher_outputs, batch, distiller, log
            )
            batch_input = batch["op_input_batch"]
        else:    # off-policy scenario
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
                outputs, teacher_outputs, batch, distiller, log
            )

        span_loss = 0.0
        if self.args.MTA_mode:
            tokenizer = distiller.student_tokenizer
            input_texts = tokenizer.batch_decode(batch_input['input_ids'], skip_special_tokens=False)
            offsets_mapping = tokenizer(input_texts, return_offsets_mapping=True, padding=True,
                                        add_special_tokens=False, return_tensors='pt')['offset_mapping']
            prases_offsets, spans_offsets, words_offsets = get_spans_offsets(input_texts, self.nlp, self.matcher)

            span_loss = compute_overall_span_loss(distiller.mta_projector_list, batch_input['attention_mask'], 
                                                outputs.hidden_states, teacher_outputs.hidden_states, 
                                                offsets_mapping, prases_offsets, spans_offsets, words_offsets, self.args)
            span_loss = self.args.w_span_loss * span_loss
            log["span_loss"] = span_loss

        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * (kd_loss + span_loss)
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, batch["label_batch"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, log)
        return loss, logging_output

    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["label_batch"]["label"]
        teacher_target = batch["teacher_label_batch"]["label"]
          
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_input = torch.where(pad_mask, batch["input_batch"]["input_ids"], torch.zeros_like(target))
        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_input = torch.where(teacher_pad_mask, batch["teacher_input_batch"][f"input_ids"], torch.zeros_like(teacher_target))
        formal_teacher_target_for_index = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        t_preds = teacher_outputs.logits.argmax(-1)
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target_for_index).detach()
        tea_preds_embeds = tea_embed_tokens(t_preds).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_preds_embeds = tea_preds_embeds / tea_preds_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.query_projector(stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # teacher space
        if distiller.part_teacher_head_pinv is not None:
            stu_lmhead = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lmhead = stu_lmhead[:, distiller.student_overlap_token_ids]
            s2t_proj = stu_lmhead @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_proj
        else:
            stu_v_hiddens = distiller.s2t_projectors(hiddens).float()  # n x d x d x D -> n x D

        tea_v_hiddens = distiller.t2s_projectors(norm_teacher_hiddens + norm_tea_preds_embeds)  # m x D x D x d -> m x d

        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        # student space
        t2s_weight = torch.softmax(align, -1).to(hiddens)      
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)  # n x m x m x d -> n x d
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]
  
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum() / pad_mask.sum()
        t2s_acc_ratio = t2s_acc_mask.sum() / pad_mask.sum()
        log["t2s_acc"] = t2s_acc
        log["t2s_acc_ratio"] = t2s_acc_ratio

        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, target, reduction="sum"
        )[0] / batch["label_batch"]["loss_denom"]
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() / batch["label_batch"]["loss_denom"]

        log["t2s_ce_loss"] = t2s_ce_loss

        # teacher space
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # m x n x n x D -> m x D
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["label_batch"]["loss_denom"]

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
      

    def compute_on_policy_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["op_label_batch"]["label"]
        teacher_target = batch["op_teacher_label_batch"]["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, batch["op_input_batch"]["input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target_for_index = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        t_preds = teacher_outputs.logits.argmax(-1)
        formal_teacher_input = torch.where(teacher_pad_mask, batch["op_teacher_input_batch"][f"input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target_for_index).detach()
        tea_preds_embeds = tea_embed_tokens(t_preds).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_preds_embeds = tea_preds_embeds / tea_preds_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.query_projector(stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # teacher space
        if distiller.part_teacher_head_pinv is not None:
            stu_lmhead = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lmhead = stu_lmhead[:, distiller.student_overlap_token_ids]
            s2t_proj = stu_lmhead @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_proj
        else:
            stu_v_hiddens = distiller.s2t_projectors(hiddens).float()  # n x d x d x D -> n x D

        tea_v_hiddens = distiller.t2s_projectors(norm_teacher_hiddens + norm_tea_preds_embeds)  # m x D x D x d -> m x d

        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
       
        align = align + (1.0 - align_mask) * (-100000)

        # student space
        t2s_weight = torch.softmax(align, -1).to(hiddens)      
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)  # n x m x m x d -> n x d
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]
        
        # CMA for on-policy distillation is a little different, since there is no gold label for projected teacher distribution
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
        assert t_preds.shape == t2s_logits.shape[:2]
        t_preds_as_label = []
        align_ratio = []
        for i in range(t_preds.shape[0]):
            indices_t_preds = torch.where(t_preds[i] != -100)[0]
            if indices_t_preds.shape[0] == 0:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=t2s_logits.device))
                align_ratio.append(1.0)
                continue
            indices_t_target = torch.where(teacher_target[i] != -100)[0]
            indices_s_target = torch.where(target[i] != -100)[0]
            
            cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1]+1]
            cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1]+1]
            cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target)

            cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1]+1]
            cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target)

            align_t_idx, align_s_idx = align_sequences(
                cur_t_target_tokens, 
                cur_s_target_tokens,
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            cur_align_ratio = len(align_s_idx) / len(cur_s_target)
            align_ratio.append(cur_align_ratio)

            cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
            cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
            for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
                tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([cur_t_preds[_t_idx]])
                try:
                    tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                    if len(tmp) == 1:
                        cur_t_preds_as_label_2[_s_idx] = tmp[0]
                    else:
                        cur_t_preds_as_label_2[_s_idx] = -100
                except:
                    cur_t_preds_as_label_2[_s_idx] = -100

            assert len(cur_t_preds_as_label_2) == len(cur_s_target)
            cur_t_preds_as_label_3 = target[i][indices_s_target[-1]+1:].cpu().tolist()
            
            cur_t_preds_as_label = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
            cur_t_preds_as_label = cur_t_preds_as_label[:t2s_logits.shape[1]]
            t_preds_as_label.append(torch.tensor(cur_t_preds_as_label, device=t2s_logits.device))

        t_preds_as_label = torch.cat(t_preds_as_label, dim=0).reshape(-1, t2s_logits.shape[1])
        log["align_ratio"] = torch.tensor(align_ratio, device=t2s_logits.device).mean()

        # calculate t2s_ce_loss only on aligned tokens from both sequences
        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds_as_label, reduction="sum"
        )[0] / batch["op_label_batch"]["loss_denom"]

        t2s_acc_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        t2s_acc = (t2s_acc_mask * t_preds_as_label.ne(-100)).sum() / max(1e-3, t_preds_as_label.ne(-100).sum())
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_agreement"] = t2s_acc

        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() / max(1e-3, t_preds_as_label.ne(-100).sum())

        # teacher space
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # m x n x n x D -> m x D
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)

        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["op_label_batch"]["loss_denom"]

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
