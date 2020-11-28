import math
import time

import torch
import torch.nn.functional as F
import torchtext
from torch import nn
from torchtext.data.utils import get_tokenizer


#
# Model Configuration
#

class BertConfig():
    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=1,
                 num_attention_heads=2,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.2,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 output_attentions=True,
                 output_hidden_states=False,
                 **kwargs):
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states


#
# Layer Normalization
#


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


#
# Self Attention
#


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # input: hidden size
        # output: all of head size
        '''
        w_h_1 w_h_2 p_h_1 p_h_2 tt_h_1 tt_h_2
        w_h_1 w_h_2 p_h_1 p_h_2 tt_h_1 tt_h_2
        w_h_1 w_h_2 p_h_1 p_h_2 tt_h_1 tt_h_2
        '''
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_score(self, x):
        # a = torch.rand([1, 2, 768])
        # a.size()[:-1]
        # torch.Size([1, 2])

        # torch.Size([1, 2])
        # torch.Size([1, 2]) + (self.num_attention_heads, self.attention_head_size)
        # (1, 2, 12, 64)
        new_x__shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        print('new_x__shape')
        print(new_x__shape)
        x = x.view(*new_x__shape)
        # batch size, attention_head, seq_len, head_size
        return x.permute(0, 2, 1, 3)  # torch.Size([1, 12, 2, 64])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        # input [batch_size, seq_len, hidden_dim]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.key(hidden_states)

        query_layer = self.transpose_for_score(mixed_query_layer)
        key_layer = self.transpose_for_score(mixed_key_layer)
        value_layer = self.transpose_for_score(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        # batch size, head no, seq_len, hidden hidden vec
        # batch size, head no, hidden hidden vec, seq_len

        # torch.Size([1, 12, 2, 64]) * torch.Size([1, 12, 64, 2])
        # output : torch.Size([1, 12, 2, 2])
        # calc attention score in header-wise
        # matrix multiplication essentially applying one matrix to another, and then average each group
        # swap item in -1, -2th indices
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # >>> math.sqrt(768)
        # 27.712812921102035
        # divide by attention_head_size
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention mask -> block to see future sentence
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_score ([1, 12, 2, 2]) + ([1, 1, 1, 2]) = ([1, 12, 2, 2])
        attention_scores = attention_scores + attention_mask

        # dim=-1 -> calc softmax using that specific dimension
        # batch size, attention_head, seq_len, head_size
        # attention prob per word in per sequence
        attention_prob = nn.Softmax(dim=-1)(attention_scores)  # -> ([1, 12, 2, 2])

        # This is actually dropping out entire tokens to attend to, which might
        # seems a bit unusual but is taken from the original Transformer paper.
        # drop `attention_weight` for some random token
        attention_probs = self.dropout(attention_prob)

        # TODO: check this variable is used in Decoder parts for prevent to seeing future sentences
        if head_mask is not None:
            attention_probs = attention_prob * head_mask

        # a value that attention softmax prob is applied
        # [1, 12, 2, 2] * [1, 12, 2, 64] = 1, 12, 2, 64 #TODO: how matrix multiplication made?
        context_layer = torch.matmul(attention_probs, value_layer)

        # permute -> dimension reodering
        # contiguous -> save variable in contiguois space for faster future operations
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 1, 2, 12, 64

        # reshape output tensors
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 1, 2, 768
        context_layer = context_layer.view(*new_context_layer_shape)  # -> 1, 2, 768

        # TODO: is the context layer name is for cls_token?
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        return outputs


#
# Bert Self Output
#
# TODO: add TC


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


#
# Bert Attention
#


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    # todo: delete unnecessary head by their index. It is omitted in the original paper, so we skip.
    def prune_heads(self, heads):
        if len(heads) == 0:
            return

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


#
# Bert Intermediate
#

class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertIntermediate, self).__init__()
        # 768 -> 3072 for increase model capacity
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertOutput, self).__init__()
        # 3072 -> 768 for bottleneck layer
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


#
# Bert Layer
#

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


#
# Bert Encoder
#

class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states  # TODO: 값은?
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        # accululate representation from each layer
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                # accumulate hidden state
                all_hidden_states = all_hidden_states + (hidden_states,)
            # TODO: what head_mask value need to be?
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            # cache for use as input for next interation
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                # accumulate attention prob
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        # accumulate to final hiden state (to the output of last bert layer)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


#
# Bert Embedding
#

class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertEmbedding, self).__init__()
        # hidden layer per vocab size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # TODO: transformers library seems assume positional encoding only need to be distinct linear layer
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        # get size of given index
        # a = torch.tensor([[1, 2, 4], [1, 2, 3]])
        # a.size(1) -> 3
        seq_length = input_ids.size(1)

        if position_ids is None:
            # make position id starting from zero
            # torch.arange(5) -> tensor([ 0,  1,  2,  3,  4])
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        if token_type_ids is None:
            # make zero-like tensor for input_id
            # input = torch.empty(2, 3)
            # torch.zeros_like(input)
            # tensor([[ 0.,  0.,  0.],
            #         [ 0.,  0.,  0.]])
            token_type_ids = torch.zeros_like(input_ids)

        # set device for tensor allocation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # lookup each element in the input_ids as size of hidden layer
        word_embeddings = self.word_embeddings(input_ids).to(device)
        position_embeddings = self.position_embeddings(position_ids).to(device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids).to(device)

        '''
        w_h_1 w_h_2 p_h_1 p_h_2 tt_h_1 tt_h_2
        w_h_1 w_h_2 p_h_1 p_h_2 tt_h_1 tt_h_2
        w_h_1 w_h_2 p_h_1 p_h_2 tt_h_1 tt_h_2
        '''
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        # layer normalization regulatize data distribution
        # to accelerate learning time by helps to escpate plateau gradient points
        embeddings = self.LayerNorm(embeddings)
        # dropout some embedding in training time
        embeddings = self.dropout(embeddings)
        torch.set_printoptions(edgeitems=3)

        return embeddings


#
# Bert Pooler
#

class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        # TODO: activation function for tensor.. 어떻게 동작하는지 구체적으로 확인 필요
        pooled_output = self.activation(pooled_output)
        return pooled_output


#
# Bert Model
#

class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.config = config

    # TODO: TBA
    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            # filling to 1
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            # filling to 0
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        # Returns a new tensor with a dimension of size one inserted at the specified position.

        # torch.unsqueeze(input, dim) → Tensor
        # add dimension to specified dimension
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # convert data type to parameters's type
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # want to attend : 1.0
        # not want to attend : 0.0
        # below op convert `not want to attend` value to -10000

        # change `1` value to 0, otherwise amplify it to -10000
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]

        # head_mask omitted
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward

        # create header with `None` to number of `config.num_hidden_layers`
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)

        sequence_output = encoder_outputs[0]  # 0th index contains attention weighted matrix
        pooled_output = self.pooler(sequence_output)  # cls token matrix

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attention)


#
# Bert For Language Model
#

class BertForLM(nn.Module):
    def __init__(self, config: BertConfig, cls_output_num, device):
        super(BertForLM, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_output_num = cls_output_num
        self.device = device
        self.cls_head = nn.Linear(config.hidden_size, cls_output_num)

    def forward(self, input_ids, labels=None, token_type_ids=None, attention_mask=None,
                selected_indices=None, **kwargs):
        bert_outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  # cls token

        sequence_output = self.dropout(sequence_output)
        # however, for language modeling task, we do not use pooled_output
        # because task output is predicting next sequence, not abstract enture sentence
        # pooled_output = self.dropout(pooled_output)

        logits = self.cls_head(sequence_output)
        predictions = logits.argmax(-1)
        loss = F.cross_entropy(logits.view(-1, self.cls_output_num), labels)

        outputs = {
            "logits": logits,
            "predictions": predictions,
            "loss": loss
        }

        return outputs


#
# Data Preprocessing
#

TEXT = torchtext.data.Field(tokenize=get_tokenizer("spacy"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


#
# Train-Eval loop
#


model = BertForLM(BertConfig(), ntokens).to(device)

# crossEntropy as loss function
# AdamW as optimizer (gradient descent method)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1E-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0)


def train():
    model.train()  # Turn on the train mode (gradient update)
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)  # get batch of training data
        optimizer.zero_grad()  # cleanup accumulated gradient https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        output = model(data, targets)  # forward model
        loss = output['loss']
        loss.backward()  # perform backprop using loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # regularize gradient
        optimizer.step()  # stepping optimizer

        total_loss += loss.item()  # accumulate loss value
        log_interval = 50  # logging in unit interval
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data, targets)
            total_loss += len(data) * output['loss']
    return total_loss / (len(data_source) - 1)


#
# Running Train-Eval loop
#

# training loop
best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

# eval loop
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
