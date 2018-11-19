
import sys
import os
import time
import collections

import tensorflow as tf


# TODO: Make an argument (dataset arg?).
MAX_DECODER_LENGTH = 100
STOP_SYMBOL = 1


def expand_by_beam(v, beam_size):
    # v: batch size x ...
    # output: (batch size * beam size) x ...

    tile_multiples = tf.concat([[1, beam_size], [1] * (tf.rank(v) - 1)], axis=0)
    tiled = tf.tile(tf.expand_dims(v, 1), tile_multiples)

    shape = tf.concat([[1], tf.slice(tf.shape(v), [1], [-1])], axis=0)
    return tf.reshape(tiled, shape)


class BeamSearchMemory(object):
    '''Batched memory used in beam search. Wrap input to LSTMDecoder (graph embeddings) with this.'''

    def __init__(self, value):
        self.value = value

    def expand_by_beam(self, beam_size):
        '''Return a copy of self where each item has been replicated
        `beam_search` times.'''
        return BeamSearchMemory(expand_by_beam(self.value, beam_size))

class BeamSearchState(collections.namedtuple('BeamSearchState', ['h', 'c'])):
    '''Batched recurrent state used in beam search. Wrap hidden state of LSTMDecoder with this.'''

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch_size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        selected = []
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = tf.reshape(v, [2, batch_size, -1, *tf.shape(v)[2:]])
            # result: 2 x indices.shape[1] x num pairs x hidden
            # TODO: make this work with tensorflow
            #selected.append(v[(slice(None), ) + tuple(indices)])
            selected.append(tf.slice(v, [(slice(None), ) + tuple(indices)]))

        return BeamSearchState(*selected)


BeamSearchResult = collections.namedtuple('BeamSearchResult', ['sequence', 'total_log_prob', 'log_probs'])


def beam_search(batch_size,
                init_state,
                memory,
                decode_fn,
                beam_size,
                max_decoder_length=MAX_DECODER_LENGTH,
                #return_attention=False,
                return_beam_search_result=False):
    # enc: batch size x hidden size
    # memory: batch size x sequence length x hidden size
    prev_preds = tf.zeros([batch_size], dtype=tf.int32)
    prev_probs = tf.zeros([batch_size, 1])
    prev_hidden = init_state
    finished = [[] for _ in range(batch_size)]
    result = [[BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0)
               for _ in range(beam_size)] for _ in range(batch_size)]
    batch_finished = [False for _ in range(batch_size)]

    # b_idx: 0, ..., 0, 1, ..., 1, ..., b, ..., b
    # where b is the batch size, and each group of numbers has as many elements
    # as the beam size.
    b_idx_untiled = tf.expand_dims(tf.range(0, batch_size), 1)
    b_idx = tf.reshape(tf.tile(b_idx_untiled, [1, beam_size]), [-1])

    prev_memory = memory.expand_by_beam(beam_size)
    #attn_list = [] if return_attention else None
    for step in range(max_decoder_length):
        logits, hidden = decode_fn(prev_preds, prev_hidden,
                                   prev_memory if step > 0 else
                                   memory, single_step=True)

        logit_size = tf.shape(logits)[1]
        # log_probs: batch size x beam size x vocab size
        log_probs = tf.reshape(tf.nn.log_softmax(logits, dim=-1), [batch_size, -1, logit_size])
        total_log_probs = log_probs + tf.expand_dims(prev_probs, 2)
        # log_probs_flat: batch size x beam_size * vocab_size
        log_probs_flat = tf.reshape(total_log_probs, [batch_size, -1])
        # indices: batch size x beam size
        # Each entry is in [0, beam_size * vocab_size)
        prev_probs, indices = tf.nn.top_k(log_probs_flat, k=min(beam_size, tf.shape(log_probs_flat)[1]))
        # prev_preds: batch_size * beam size
        # Each entry indicates which pred should be added to each beam.
        prev_preds = tf.reshape(tf.mod(indices, logit_size), [-1])
        # This takes a lot of time... about 50% of the whole thing.
        # k_idx: batch size x beam size
        # Each entry is in [0, beam_size), indicating which beam to extend.
        k_idx = tf.div(indices, logit_size)
        idx = tf.stack([b_idx, tf.reshape(k_idx, [-1])])
        # prev_hidden: (batch size * beam size) x hidden size
        # Contains the hidden states which produced the top-k (k = beam size)
        # preds, and should be extended in the step.
        # TODO: make this work with tensorflow
        prev_hidden = hidden.select_for_beams(batch_size, idx)

        prev_result = result
        result = [[] for _ in range(batch_size)]
        can_stop = True

        for batch_id in range(batch_size):
            # print(step, finished[batch_id])
            if len(finished[batch_id]) >= beam_size:
                # If last in finished has bigger log prob then best in topk, stop.
                fin = tf.cond(finished[batch_id][-1].total_log_prob > prev_probs[batch_id, 0],
                        lambda: True, lambda: False)

                if fin:
                    batch_finished[batch_id] = True
                    continue

                #if finished[batch_id][-1].total_log_prob > prev_probs[batch_id, 0]:
                #    batch_finished[batch_id] = True
                #    continue

            for idx in range(beam_size):
                pred = tf.mod(indices[batch_id, idx], logit_size)
                kidx = k_idx[batch_id, idx]
                # print(step, batch_id, idx, 'pred', pred, kidx, 'prev', prev_result[batch_id][kidx], prev_probs.data[batch_id][idx])
                if step == max_decoder_length - 1: # or tf.equal(pred, STOP_SYMBOL):  # 1 == </S>
                    finished[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence + [pred],
                        total_log_prob=prev_probs[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs[batch_id, kidx, pred]]))
                    #result[batch_id].append(BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0))
                    #prev_probs.data[batch_id][idx] = float('-inf') # this beam reached EOS so there's no way to add more to it.
                else:
                    result[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence + [pred],
                        total_log_prob=prev_probs[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs[batch_id, kidx, pred]]))
                    can_stop = False # If a single beam is not at EOS then we must go to next step
            if len(finished[batch_id]) >= beam_size:
                # Sort and clip.
                finished[batch_id] = sorted(
                    finished[batch_id], key=lambda x: -x.total_log_prob)[:beam_size]
        if can_stop:
            break

    for batch_id in range(batch_size):
        # If there is deficit in finished, fill it in with highest probable results.
        if len(finished[batch_id]) < beam_size:
            i = 0
            while i < beam_size and len(finished[batch_id]) < beam_size:
                if result[batch_id][i]:
                    finished[batch_id].append(result[batch_id][i])
                i += 1

    if not return_beam_search_result:
        for batch_id in range(batch_size):
            finished[batch_id] = [x.sequence for x in finished[batch_id]]

    #if return_attention:
    #    # all elements of attn_list: (batch size * beam size) x input length
    #    attn_list[0] = expand_by_beam(attn_list[0], beam_size)
    #    # attns: batch size x bean size x out length x inp length
    #    attns = torch.stack(
    #            [attn.view(batch_size, -1, attn.size(1)) for attn in attn_list],
    #            dim=2)
    #    return finished, attns
    return finished
