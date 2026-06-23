import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded_text, text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded_text, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc(hidden)
        return output


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence_output, lengths):
        scores = self.attention(sequence_output).squeeze(-1)
        batch_size, max_len = sequence_output.size(0), sequence_output.size(1)
        mask = (
            torch.arange(max_len, device=sequence_output.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence_output).squeeze(1)
        return context, weights


class ImprovedGRUClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout=0.5,
        num_extra_features=0,
        use_feature_fusion=False,
    ):
        super(ImprovedGRUClassifier, self).__init__()
        self.use_feature_fusion = use_feature_fusion
        self.num_extra_features = num_extra_features
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = AttentionPooling(hidden_dim * 2)
        pooled_dim = hidden_dim * 2 * 2
        self.layer_norm = nn.LayerNorm(pooled_dim)
        mlp_input_dim = pooled_dim
        if use_feature_fusion and num_extra_features > 0:
            mlp_input_dim += num_extra_features
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded_text, text_lengths, extra_features=None):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded_text, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_embedded)
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        attention_context, attention_weights = self.attention(sequence_output, text_lengths)
        batch_size, max_len, hidden_size = sequence_output.size()
        mask = (
            torch.arange(max_len, device=sequence_output.device).unsqueeze(0)
            < text_lengths.unsqueeze(1)
        )
        mask = mask.unsqueeze(-1).float()
        masked_output = sequence_output * mask
        sum_output = masked_output.sum(dim=1)
        mean_output = sum_output / text_lengths.unsqueeze(1).float()
        pooled = torch.cat([attention_context, mean_output], dim=1)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        if self.use_feature_fusion and extra_features is not None:
            pooled = torch.cat([pooled, extra_features], dim=1)
        output = self.classifier(pooled)
        return output, attention_weights
