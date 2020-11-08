from modeling_bert import *


class TestModelingBert():
    def test_BertEmbedding(self):
        embedding = BertEmbedding(BertConfig())
        print(embedding.forward(torch.tensor([[2, 2]], dtype=torch.long)))

    def test_BertSelfAttention(self):
        self_attention = BertSelfAttention(BertConfig())
        input_hidden_state = torch.randn(1, 1, 768)  # batch, words, embeddings per word
        attention_mask = torch.randn(1, 12, 1, 1)  # attention mask for each batch
        print(self_attention.forward(input_hidden_state, attention_mask))

    def test_BertSelfOutput(self):
        self_output = BertSelfOutput(BertConfig())
        input_hidden_state = torch.randn(1, 1, 768)
        input_tensor = torch.randn(1, 1, 768)
        print(self_output.forward(input_hidden_state, input_tensor))

    def test_BertAttention(self):
        attention = BertAttention(BertConfig())
        input_hidden_state = torch.randn(1, 1, 768)  # batch, words, embeddings per word
        attention_mask = torch.randn(1, 12, 1, 1)  # attention mask for each batch
        print(attention.forward(input_hidden_state, attention_mask))

    def test_BertIntermediate(self):
        intermediate_result = BertIntermediate(BertConfig())
        input_hidden_state = torch.randn(1, 1, 768)  # batch, words, embeddings per word
        print(intermediate_result.forward(input_hidden_state))

    def test_BertOutput(self):
        bert_output = BertOutput(BertConfig())
        input_hidden_state = torch.randn(1, 1, 3072)  # result after bert self output
        input_tensor = torch.randn(1, 1, 768)  # residual input
        print(bert_output(input_hidden_state, input_tensor))

    def test_BertLayer(self):
        bert_layer = BertLayer(BertConfig())
        input_hidden_state = torch.randn(1, 1, 768)  # batch, words, embeddings per word
        attention_mask = torch.randn(1, 12, 1, 1)  # attention mask for each batch
        print(bert_layer(input_hidden_state, attention_mask))

    def test_BertEncoder(self):
        config = BertConfig()
        bert_encoder = BertEncoder(config)
        input_hidden_state = torch.randn(1, 1, 768)
        attention_mask = torch.randn(1, 12, 1, 1)
        head_mask = [None] * config.num_hidden_layers
        print(bert_encoder(input_hidden_state, attention_mask, head_mask))

    def test_BertModel(self):
        bert_model = BertModel(BertConfig())
        input = torch.tensor([[1, 1]])
        # input : batch of words
        result = bert_model(input)
        # output : tuple result of 768 (hidden size)
        print(result)

    def testBertForLM(self):
        bert_for_nlu = BertForLM(BertConfig())
        input = torch.tensor([[1, 1]])  # softmax output
        labels = torch.LongTensor([0.99])  # cross entropy range : ~ 0.99
        result = bert_for_nlu(input, labels)
        print(result)
