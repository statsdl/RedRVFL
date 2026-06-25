import torch
import torch.nn as nn


class RecRVFL(nn.Module):

    def __init__(self, l, features, raw_features, nodes, lamb, input_scale, layers=1, device='cpu'):
        super(RecRVFL, self).__init__()

        # init params
        self.l = l
        self.layers =layers
        self.features = features
        self.raw_features = raw_features

        self.nodes = nodes
        self.lamb = lamb
        self.input_scale = input_scale
        self.d = device

        if self.l == 0:
            self.layer = nn.Sequential(nn.LSTM(self.features, int(self.nodes), num_layers=self.layers),)
            # self.layer = nn.Sequential(nn.GRU(self.features, int(self.nodes), num_layers=self.layers),)
        else:
            self.layer = nn.Sequential(nn.LSTM(self.features + self.raw_features, int(self.nodes), num_layers=self.layers),)
            
        # Apply weight initialization to the LSTM layer
        self.layer.apply(lambda m: self.__init_lstm_weights__(m, input_scale))    
            
        self.linear = nn.Sequential(nn.Linear(int(self.nodes), int(self.nodes)),)
        self.relu = nn.Sigmoid()
        self.linear.apply(self.__init_weights__)
        
        self.output = nn.Sequential(
            nn.Linear(int(self.nodes) + self.raw_features + 1, 1))


    def __init_lstm_weights__(self, m, input_scale):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.uniform_(param.data, -input_scale, input_scale)
                elif 'bias' in name:
                    param.data.fill_(0)
                    
                    
    def __init_weights__(self, m):

        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-self.input_scale, self.input_scale)
            m.bias.data.fill_(0)

    def init_weight(self, X, y, X_raw):
        n_sample = X.shape[0]
        encoding = self.transform(X_raw, X)
        merged = torch.cat((encoding, X_raw, torch.ones((n_sample, 1)).to(self.d)), dim=-1)  # for direct links
        n_features = merged.shape[1]

        if n_features < n_sample:
            # prime space equation
            # (I.lamb + D^T.D)^-1 . D^T . Y
            beta = torch.mm(
                torch.mm(torch.inverse(torch.eye(merged.shape[1]).to(self.d)*self.lamb + torch.mm(merged.T, merged)),
                          merged.T), y)
        else:
            # dual space equation
            # D^T . (lamb.I + D.D^T)^-1 . Y
            beta = torch.mm(merged.T, torch.mm(
                torch.inverse(torch.eye(merged.shape[0]).to(self.d)*self.lamb + torch.mm(merged, merged.T)), y))

        self.output[0].weight = nn.Parameter(beta.T)
        self.output[0].bias.data.fill_(0)

    def forward(self, X_raw, X = None):
        if self.l == 0:
            encoding,_ = self.layer(X_raw)
        else:
            encoding,_ = self.layer(torch.cat((X, X_raw), dim=1))
        encoding = self.relu(encoding)
        encoding = self.linear(encoding)
        encoding = self.relu(encoding)
        merged = torch.cat((encoding, X_raw, torch.ones((X_raw.shape[0], 1)).to(self.d)), dim=1)
        out = self.output(merged)
        return out

    def transform(self, X_raw, X = None):
        if self.l == 0:
            encoding,_ = self.layer(X_raw)
        else:
            encoding,_ = self.layer(torch.cat((X, X_raw), dim=1))
        encoding = self.relu(encoding)
        
        encoding = self.linear(encoding)
        encoding = self.relu(encoding)
        return encoding