import torch 
from torch import nn
from torch.functional import F
from torch.distributed import rpc
from torch.distributed.rpc import RRef

class SubNetwork(nn.Module):
    def __init__(self):
        super(SubNetwork,self).__init__()
        pass

    def _parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class Encoder(SubNetwork):
    def __init__(self,latent_dim,input_dim,gpu=False):
        super(Encoder,self).__init__()
        if gpu:
            self.devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        else:
            self.devices = ['cpu','cpu']
        self.fc1 = nn.Linear(input_dim,128).to(self.devices[0])
        self.fc2 = nn.Linear(128, latent_dim).to(self.devices[1])
    
    def forward(self,x):
        x = F.relu(self.fc1(x.to(self.devices[0])))
        x = self.fc2(x.to(self.devices[1]))
        return x.cpu()

class Decoder(SubNetwork):
    def __init__(self,latent_dim,output_dim,gpu=False):
        super(Decoder,self).__init__()
        if gpu:
            self.devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        else:
            self.devices = ['cpu','cpu']
        self.fc1 = nn.Linear(latent_dim,128).to(self.devices[0])
        self.fc2 = nn.Linear(128, output_dim).to(self.devices[1])
    
    def forward(self,x):
        x = F.relu(self.fc1(x.to(self.devices[0])))
        x = self.fc2(x.to(self.devices[1]))
        return x.cpu()

class DistAutoEncoder(nn.Module):
    def __init__(self,latent_dim,input_dim,workers):
        super(DistAutoEncoder,self).__init__()
        self.encoder_rref = rpc.remote(workers[0],Encoder,args=(latent_dim,input_dim))
        self.decoder_rref = rpc.remote(workers[1],Decoder,args=(latent_dim,input_dim))

    def forward(self, x):
        encoded = self.encoder_rref.to_here().forward(x)
        decoded = self.decoder_rref.to_here().forward(encoded)
        return decoded

    def _parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.encoder_rref.remote()._parameter_rrefs().to_here())
        remote_params.extend(self.decoder_rref.remote()._parameter_rrefs().to_here())
        return remote_params


