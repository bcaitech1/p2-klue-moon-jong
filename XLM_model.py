from torch import nn
from transformers import XLMRobertaModel
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch

class Cross_FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean', **kwargs):

        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, inputs, targets):
        loss_fct_cross = nn.CrossEntropyLoss()
        loss_cross = loss_fct_cross(inputs, targets)
        loss_fct_focal = FocalLoss()
        loss_focal = loss_fct_focal(inputs, targets)
        # 두 비율이 합쳐서 1이 되야함
        cross_rate = 0.75
        focal_rate = 0.25
        return loss_cross * cross_rate + loss_focal * focal_rate


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def compute_metrics(preds, labels):
    assert len(preds) == len(labels), 'length is not matched'
    acc = (preds == labels).mean()
    accuracy = {"acc": acc}
    return accuracy




class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class LSTMLayer(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self,output_dim, num_layers=1, dropout_rate=0.0):
        super(LSTMLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(output_dim, output_dim, num_layers)
        self.fc = nn.Linear(output_dim, 42)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.tanh(x)
        return self.fc(x[0])


class RXLMRoberta(XLMRobertaModel):
    def __init__(self, model_name, config, dropout_rate, activate_LSTM=None):
        super(RXLMRoberta, self).__init__(config)
        self.XLMRoberta = XLMRobertaModel.from_pretrained(model_name, config=config)  # Load pretrained XLMRoberta

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer1 = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer2 = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.activate_lstm = activate_LSTM
        # self.relu = nn.ReLU()
        if self.activate_lstm is not None:
            self.lstm_after_concat = LSTMLayer(config.hidden_size * 3, dropout_rate=dropout_rate)

        # self.sep_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)

        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            dropout_rate,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """

        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.XLMRoberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        # sep_h = self.entity_average(sequence_output, sep_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer1(e1_h)
        e2_h = self.entity_fc_layer2(e2_h)

        # sep_h = self.sep_fc_layer(sep_h)
        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        if self.activate_lstm is not None:
            concat_h = concat_h.view(-1, concat_h.size()[0], concat_h.size()[1])
            logits = self.lstm_after_concat(concat_h)
        else:
            logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
