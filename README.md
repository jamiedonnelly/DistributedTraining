Collection of scripts for different implementations of parallel/distributed training in **PyTorch**, and using **SLURM** scheduler for coordination and resource allocation.

The primary aim is to utilise **Model Parallelism** to train large neural networks across multiple nodes in a HPC cluster (i.e., memory footprint of the complete model exceeds the available resources on any one node). 

This is technically referred to **Pipeline Parallelism** by vertically segmenting the network into separate layers, or subnetworks, and distributing the components across nodes. The parallelism is a misnomer as training still occurrs sequentially - each node must wait for the previous node's output before beginning its forward pass. 

The key to Pipeline Parallelism using PyTorch is distributed communication via the `torch.distributed.rpc` API. Using `rpc.remote` within a superclass (i.e., whole network) definition, subnetworks can be initialised on worker nodes. The below code and example is heavily inspired from [1], [7].

**Autoencoder toy example**
_Network architecture is arbitrary and purely to illustrate the distrubuted training_ 

```
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
        x = self.fc1(x.to(self.devices[0]))
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
        x = self.fc1(x.to(self.devices[0]))
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

```

The above code shows an example Autoencoder distributed across 3 nodes, 2 worker nodes responsible for the encoder and decoder and a third for initialising the DistAutoEncoder class (with the third node acting as master, responsible for data loader, training co-ordination, etc.) 

In the above code, a base `SubNetwork` class is initialised to be inherited from in any subnetwork definition. This class is used to i) inherit from `nn.Module` and ii) create the hidden method `_parameter_rref` which returns *R*emote *Ref*erences [6] for the parameters in the subnetwork which will be later used in the distributed autograd and optimiser. 

The Encoder and Decoder consist of 2 Linear layers each (which can be distributed between 2 GPUs if desired) and the DistAutoEncoder is responsible for remotely initialising the subnetworks on each worker node, collecting *All* the parameter RRefs, and finally executing the froward pass by executing remote methods on the worker nodes. 

**Training Loop** 
The master node is responsible for initialising the model and coordination, 

```
model = DistAutoEncoder(784,28,["worker1","worker2"])
...
opt = DistributedOptimizer(optim.Adam,model._parameter_rrefs(),lr=0.01) # Distributed optimiser, utilising the hidden method _parameter_rrefs in place of `model.parameters` in a serial model
...
model.train()
for epoch in range(epochs):
    for i, (inp, lab) in enumerate(trainloader):
        with dist_autograd.context() as ctx_id: # use context manager for distributed autograd 
            print(inp.flatten(2).shape)
            return 0
            outputs = model(inp.flatten(2))
            if i % 20 == 0:
                print(f"Epoch {epoch} || Batch {i} ---- Training MSE: {loss_fn(outputs,lab).item()}")
            dist_autograd.backward(ctx_id, [loss_fn(output,inp.flatten(2))]) # Distributed backward pass
            opt.step(ctx_id) # Distributed parameter updates
```

The bulk of the training logic is the same for a distributed and serial model, the only difference is the implementation of the distributed autograd for the backward passes and the distributed optimizer for the parameter updates. A context manager, `with dist_autograd.context` is used to keep backward passes and updates synchronised across nodes. 

**Running the model**


# A list of some of the many resources I have referred to in going from having ~zero knowledge about practically implementing distributed/parallel neural network training to having a moderate understanding and getting working implementations #

=================================================================================================================================================
References/Resources:
[1] - https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html
[2] - https://pytorch.org/tutorials/intermediate/rpc_tutorial.html#
[3] - https://lambdalabs.com/blog/introduction-multi-gpu-multi-node-distributed-training-nccl-2-0
[4] - https://arxiv.org/abs/1802.09941
[5] - https://pytorch.org/docs/stable/distributed.html
[6] - https://pytorch.org/docs/stable/rpc.html
[7] - https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html
[8] - https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
[9] - https://pytorch.org/docs/stable/elastic/run.html
=================================================================================================================================================
